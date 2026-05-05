from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec 
from pinecone_text.sparse import BM25Encoder

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '').strip()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '').strip()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

texts = [doc.page_content for doc in text_chunks]
sources = [doc.metadata.get("source") for doc in text_chunks]

bm25 = BM25Encoder().fit(texts)
bm25_path = os.path.join(os.path.dirname(__file__), "bm25_encoder.json")
bm25.dump(bm25_path)
print(f"Saved BM25 encoder to: {bm25_path}")

dense_vectors = embeddings.embed_documents(texts)
sparse_vectors = bm25.encode_documents(texts)

def _safe_id(value: str) -> str:
    if not value:
        return "unknown"
    return (
        value.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

batch_size = 100
to_upsert = []
for i, (text, src, dense, sparse) in enumerate(zip(texts, sources, dense_vectors, sparse_vectors)):
    vector_id = f"{_safe_id(src)}#{i}"
    metadata = {
        "source": src,
        "text": text,
        "chunk": i,
    }
    to_upsert.append(
        {
            "id": vector_id,
            "values": dense,
            "sparse_values": sparse,
            "metadata": metadata,
        }
    )

for start in range(0, len(to_upsert), batch_size):
    batch = to_upsert[start : start + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted {min(start + batch_size, len(to_upsert))}/{len(to_upsert)} vectors")

print("Hybrid upsert complete.")