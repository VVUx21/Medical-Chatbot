from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
try:
    from pinecone_text.sparse import BM25Encoder
except ImportError:  # pragma: no cover
    BM25Encoder = None
try:
    from langchain_community.document_compressors import FlashrankRerank
except ImportError:  # pragma: no cover
    FlashrankRerank = None
try:
    from flashrank import Ranker
except ImportError:  # pragma: no cover
    Ranker = None
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Any, cast
from dotenv import load_dotenv
from src.prompt import *
import re
import os

app = Flask(__name__)

load_dotenv()

# FIX: Added .strip() to remove hidden \n or spaces from the end of keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '').strip()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '').strip()

# Set the key in environment for the libraries to pick up
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(index_name)

bm25_path = os.path.join(os.path.dirname(__file__), "bm25_encoder.json")
bm25 = None
if BM25Encoder is None:
    print("WARNING: pinecone-text is not installed; using dense-only retrieval.")
elif os.path.exists(bm25_path):
    bm25 = BM25Encoder().load(bm25_path)
else:
    print(
        f"WARNING: BM25 encoder file not found at {bm25_path}. "
        "Run `python store_index.py` to generate it for hybrid search."
    )

_flashrank_client = Ranker() if Ranker is not None else None

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
_tavily_client = None

TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "advanced"  # "basic" or "advanced"

try:
    chatModel = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_output_tokens=2048,
        streaming=True,
    )
except TypeError:
    chatModel = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_output_tokens=2048,
    )

rewriterModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0, max_output_tokens=256)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter for a medical RAG system. "
            "Given the chat history and the user's latest message, rewrite it into a standalone, "
            "search-optimized query that preserves medical meaning. "
            "Do NOT answer the question. Output only the rewritten query text."
        ),
        MessagesPlaceholder("history"),
        ("human", "Latest message: {input}\nStandalone retrieval query:"),
    ]
)

rewrite_chain = rewrite_prompt | rewriterModel | StrOutputParser()

HISTORY_MAX_TURNS = 6
_histories = {}

def _get_session_id():
    return request.remote_addr or "local"

def _get_history(session_id):
    history = _histories.get(session_id)
    if history is None:
        history = ChatMessageHistory()
        _histories[session_id] = history
    return history

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RETRIEVE_K_CANDIDATES = 10
RERANK_TOP_N = 3

def retrieve_candidates_with_scores(query, k=RETRIEVE_K_CANDIDATES):
    dense_query = embeddings.embed_query(query)
    if bm25 is not None:
        sparse_query = bm25.encode_queries(query)
        if isinstance(sparse_query, list):
            sparse_query = sparse_query[0] if sparse_query else None
        if sparse_query:
            res = pinecone_index.query(
                vector=dense_query,
                sparse_vector=cast(Any, sparse_query),
                top_k=k,
                include_metadata=True,
            )
        else:
            res = pinecone_index.query(
                vector=dense_query,
                top_k=k,
                include_metadata=True,
            )
    else:
        res = pinecone_index.query(
            vector=dense_query,
            top_k=k,
            include_metadata=True,
        )

    scored = []
    matches = getattr(res, "matches", None)
    if matches is None and isinstance(res, dict):
        matches = res["matches"] if "matches" in res else []
    matches = matches or []
    for match in matches:
        score = getattr(match, "score", None) if not isinstance(match, dict) else match.get("score")
        metadata = getattr(match, "metadata", None) if not isinstance(match, dict) else match.get("metadata")
        metadata = metadata or {}
        text = metadata.get("text", "")
        doc_meta = dict(metadata)
        doc_meta.pop("text", None)
        if score is None:
            continue
        scored.append((Document(page_content=text, metadata=doc_meta), float(score)))
    return scored

def rerank_with_flashrank(query, docs, top_n=RERANK_TOP_N):
    if not docs:
        return []
    if FlashrankRerank is None or _flashrank_client is None:
        return docs[:top_n]
    reranker = FlashrankRerank(client=_flashrank_client, top_n=top_n)
    return reranker.compress_documents(docs, query)

def tavily_search_as_docs(query, max_results=TAVILY_MAX_RESULTS):
    global _tavily_client
    if not TAVILY_API_KEY:
        return []
    if _tavily_client is None:
        try:
            from tavily import TavilyClient  # type: ignore
        except ImportError:
            print("WARNING: tavily-python is not installed; web fallback disabled.")
            return []
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        res = _tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth=TAVILY_SEARCH_DEPTH,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []

    results = res.get("results", []) if isinstance(res, dict) else []
    docs = []
    for item in results:
        content = (item.get("content") or "").strip()
        url = item.get("url")
        title = item.get("title")
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={"source": url, "title": title, "provider": "tavily"},
            )
        )
    return docs

def format_web_context(docs):
    # Keep the web context compact and attribution-friendly.
    lines = []
    for i, doc in enumerate(docs, start=1):
        src = (doc.metadata or {}).get("source")
        title = (doc.metadata or {}).get("title")
        snippet = _preview(doc.page_content, max_chars=400)
        header = f"[{i}] {title or ''}".strip()
        if src:
            header = f"{header}\nURL: {src}" if header else f"URL: {src}"
        if header:
            lines.append(header)
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip()

SCORE_HIGH_IS_BETTER = True

def evaluate_retrieval(scored_docs, high=0.60, low=0.40):
    """
    Simplified logic: 
    - If top score is above 'high', we trust it.
    - If it's below 'low', we reject it.
    - If it's in between, we call it ambiguous.
    """
    if not scored_docs:
        return "incorrect"
        
    # Extract the score from the top result
    top_score = scored_docs[0][1]
    print(f"DEBUG: Top Retrieval Score is {top_score}") # Helpful for monitoring

    if top_score >= high:
        return "correct"
    elif top_score < low:
        return "incorrect"
    else:
        return "ambiguous"

def format_scored_docs(scored_docs):
    docs = [doc for doc, _ in scored_docs]
    return format_docs(docs)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a verifier. If the draft answer is fully supported by the context, "
            "return the draft answer unchanged. If not, revise it to be grounded in the context. "
            "Return only the final answer."
        ),
        ("human", "Question: {input}\nContext: {context}\nDraft: {answer}"),
    ]
)

def is_greeting(text):
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    return cleaned.startswith(("hi", "hello", "hey"))

def _clean_rewrite(text, fallback):
    rewritten = (text or "").strip().strip('"').strip("'")
    return rewritten if rewritten else fallback

def _preview(text, max_chars=300):
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."

def answer_with_crag_self_rag(query, session_id):
    history = _get_history(session_id)

    if is_greeting(query):
        assistant_text = "Hi! Ask me a medical question and I will do my best to answer."
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return assistant_text

    # Query rewriting using short-term memory (chat history)
    prior_messages = history.messages[-2 * HISTORY_MAX_TURNS :]
    rewritten_query = rewrite_chain.invoke({"history": prior_messages, "input": query})
    rewritten_query = _clean_rewrite(rewritten_query, fallback=query)
    print(f"Rewritten query: {rewritten_query}")

    scored_docs = retrieve_candidates_with_scores(rewritten_query, k=RETRIEVE_K_CANDIDATES)
    print("Hybrid retrieval scores:", [score for _, score in scored_docs])

    print("\n--- Retrieved candidate chunks (preview) ---")
    for i, (doc, score) in enumerate(scored_docs[:4], start=1):
        src = doc.metadata.get("source") if doc.metadata else None
        print(f"[{i}] score={score:.4f} source={src}")
        print(_preview(doc.page_content, max_chars=300))
        print()
    # Lowering 'high' to 0.60 ensures your 0.68 score passes as 'correct'
    label = evaluate_retrieval(scored_docs, high=0.60, low=0.45)

    # CRAG Step 4: web-search fallback when retrieval is weak.
    if label in {"incorrect", "ambiguous"}:
        web_docs = tavily_search_as_docs(rewritten_query)
        if web_docs:
            web_docs = rerank_with_flashrank(rewritten_query, web_docs, top_n=5)
            print("--- Tavily web results used (preview) ---")
            for i, doc in enumerate(web_docs[:4], start=1):
                src = doc.metadata.get("source") if doc.metadata else None
                print(f"[{i}] source={src}")
                print(_preview(doc.page_content, max_chars=300))
                print()

            web_context = format_web_context(web_docs)
            context = (
                "WEB SEARCH RESULTS (use as supporting evidence; cite URLs when helpful):\n\n"
                + web_context
            )
            draft = (prompt | chatModel | StrOutputParser()).invoke(
                {"context": context, "input": query}
            )
            final = (reflection_prompt | chatModel | StrOutputParser()).invoke(
                {"context": context, "input": query, "answer": draft}
            )
            history.add_user_message(query)
            history.add_ai_message(final)
            return final

    if label == "ambiguous":
        assistant_text = "I might need a bit more detail to answer accurately. What specific aspect are you asking about?"
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return assistant_text
    if label == "incorrect":
        assistant_text = "I don't know based on my sources."
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return assistant_text

    candidate_docs = [doc for doc, _ in scored_docs]
    reranked_docs = rerank_with_flashrank(rewritten_query, candidate_docs, top_n=RERANK_TOP_N)

    print("--- Reranked chunks used for context (preview) ---")
    for i, doc in enumerate(reranked_docs[:4], start=1):
        src = doc.metadata.get("source") if doc.metadata else None
        print(f"[{i}] source={src}")
        print(_preview(doc.page_content, max_chars=300))
        print()

    context = format_docs(reranked_docs)
    draft = (prompt | chatModel | StrOutputParser()).invoke(
        {"context": context, "input": query}
    )
    final = (reflection_prompt | chatModel | StrOutputParser()).invoke(
        {"context": context, "input": query, "answer": draft}
    )
    history.add_user_message(query)
    history.add_ai_message(final)
    return final

def answer_with_crag_self_rag_stream(query, session_id):
    history = _get_history(session_id)

    def _stream_text(text):
        yield text

    if is_greeting(query):
        assistant_text = "Hi! Ask me a medical question and I will do my best to answer."
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return _stream_text(assistant_text)

    prior_messages = history.messages[-2 * HISTORY_MAX_TURNS :]
    rewritten_query = rewrite_chain.invoke({"history": prior_messages, "input": query})
    rewritten_query = _clean_rewrite(rewritten_query, fallback=query)
    print(f"Rewritten query: {rewritten_query}")

    scored_docs = retrieve_candidates_with_scores(rewritten_query, k=RETRIEVE_K_CANDIDATES)
    print("Hybrid retrieval scores:", [score for _, score in scored_docs])

    print("\n--- Retrieved candidate chunks (preview) ---")
    for i, (doc, score) in enumerate(scored_docs[:4], start=1):
        src = doc.metadata.get("source") if doc.metadata else None
        print(f"[{i}] score={score:.4f} source={src}")
        print(_preview(doc.page_content, max_chars=300))
        print()

    label = evaluate_retrieval(scored_docs, high=0.60, low=0.45)

    # Web fallback first (CRAG Step 4)
    if label in {"incorrect", "ambiguous"}:
        web_docs = tavily_search_as_docs(rewritten_query)
        if web_docs:
            web_docs = rerank_with_flashrank(rewritten_query, web_docs, top_n=5)
            print("--- Tavily web results used (preview) ---")
            for i, doc in enumerate(web_docs[:4], start=1):
                src = doc.metadata.get("source") if doc.metadata else None
                print(f"[{i}] source={src}")
                print(_preview(doc.page_content, max_chars=300))
                print()

            web_context = format_web_context(web_docs)
            context = (
                "WEB SEARCH RESULTS (use as supporting evidence; cite URLs when helpful):\n\n"
                + web_context
            )

            draft = (prompt | chatModel | StrOutputParser()).invoke({"context": context, "input": query})
            final_chain = reflection_prompt | chatModel | StrOutputParser()

            def gen():
                buffer = []
                try:
                    for chunk in final_chain.stream({"context": context, "input": query, "answer": draft}):
                        buffer.append(chunk)
                        yield chunk
                except Exception:
                    final_text = final_chain.invoke({"context": context, "input": query, "answer": draft})
                    yield final_text
                    buffer = [final_text]
                final_text = "".join(buffer)
                history.add_user_message(query)
                history.add_ai_message(final_text)

            return gen()

    if label == "ambiguous":
        assistant_text = "I might need a bit more detail to answer accurately. What specific aspect are you asking about?"
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return _stream_text(assistant_text)

    if label == "incorrect":
        assistant_text = "I don't know based on my sources."
        history.add_user_message(query)
        history.add_ai_message(assistant_text)
        return _stream_text(assistant_text)

    candidate_docs = [doc for doc, _ in scored_docs]
    reranked_docs = rerank_with_flashrank(rewritten_query, candidate_docs, top_n=RERANK_TOP_N)

    print("--- Reranked chunks used for context (preview) ---")
    for i, doc in enumerate(reranked_docs[:4], start=1):
        src = doc.metadata.get("source") if doc.metadata else None
        print(f"[{i}] source={src}")
        print(_preview(doc.page_content, max_chars=300))
        print()

    context = format_docs(reranked_docs)
    draft = (prompt | chatModel | StrOutputParser()).invoke({"context": context, "input": query})
    final_chain = reflection_prompt | chatModel | StrOutputParser()

    def gen():
        buffer = []
        try:
            for chunk in final_chain.stream({"context": context, "input": query, "answer": draft}):
                buffer.append(chunk)
                yield chunk
        except Exception:
            final_text = final_chain.invoke({"context": context, "input": query, "answer": draft})
            yield final_text
            buffer = [final_text]
        final_text = "".join(buffer)
        history.add_user_message(query)
        history.add_ai_message(final_text)

    return gen()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    session_id = _get_session_id()
    response = answer_with_crag_self_rag(msg, session_id=session_id)
    print("Response: ", response)
    return str(response)

@app.route("/get_stream", methods=["POST"])
def chat_stream():
    msg = request.form["msg"]
    print(f"User Input (stream): {msg}")
    session_id = _get_session_id()

    def generate():
        try:
            for chunk in answer_with_crag_self_rag_stream(msg, session_id=session_id):
                yield chunk
        except Exception as e:
            print(f"Streaming error: {e}")
            yield "\n[Streaming error]"

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
