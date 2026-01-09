FROM python:3.10-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
&& pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim-bookworm

WORKDIR /app

# Runtime-only dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages only
COPY --from=builder /usr/local /usr/local

COPY . .

EXPOSE 8080

CMD ["python3", "app.py"]
