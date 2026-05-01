
#endpoint = "https://gaurav-testing.openai.azure.com/"
#model_name = "gpt-5.4-mini"
#deployment = "gpt-5.4-mini"
# main.py
import os
import uuid
import shutil
import math
import configparser
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from azure.cosmos import CosmosClient, PartitionKey
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI


# ---------------------------------------------------------------------------
# Config — loaded from config.ini, with environment variables as overrides
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("APP_CONFIG", "config.ini")
config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def get_config(section: str, key: str, env_key: Optional[str] = None, default: Optional[str] = None) -> str:
    """Read config from env first, then config.ini, then default."""
    env_name = env_key or key.upper()
    value = os.environ.get(env_name)
    if value is None and config.has_option(section, key):
        value = config.get(section, key)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(
            f"Missing required config value [{section}] {key}. "
            f"Set it in {CONFIG_PATH} or export {env_name}."
        )
    return value.strip()


def require_real_value(name: str, value: str) -> str:
    """Fail fast when config.ini still contains a placeholder."""
    if value.startswith("paste-your-") or "your-cosmos-account" in value:
        raise RuntimeError(f"Replace the placeholder value for {name} in {CONFIG_PATH}.")
    return value


def get_config_int(section: str, key: str, env_key: Optional[str] = None, default: int = 0) -> int:
    return int(get_config(section, key, env_key, str(default)))


def get_config_float(section: str, key: str, env_key: Optional[str] = None, default: float = 0.0) -> float:
    return float(get_config(section, key, env_key, str(default)))


def get_config_bool(section: str, key: str, env_key: Optional[str] = None, default: bool = False) -> bool:
    value = get_config(section, key, env_key, str(default)).lower()
    return value in {"1", "true", "yes", "on"}


AZURE_OPENAI_ENDPOINT = require_real_value(
    "azure_openai.endpoint",
    get_config("azure_openai", "endpoint", "AZURE_OPENAI_ENDPOINT").rstrip("/"),
)
AZURE_OPENAI_KEY      = require_real_value(
    "azure_openai.key",
    get_config("azure_openai", "key", "AZURE_OPENAI_KEY"),
)
AZURE_OPENAI_DEPLOY   = get_config("azure_openai", "deployment", "AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
AZURE_API_VERSION     = get_config("azure_openai", "api_version", "AZURE_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_TIMEOUT  = get_config_float("azure_openai", "timeout", "AZURE_OPENAI_TIMEOUT", 60)

COSMOS_ENDPOINT  = require_real_value("cosmos.endpoint", get_config("cosmos", "endpoint", "COSMOS_ENDPOINT"))
COSMOS_KEY       = require_real_value("cosmos.key", get_config("cosmos", "key", "COSMOS_KEY"))
COSMOS_DATABASE  = get_config("cosmos", "database", "COSMOS_DATABASE", "rag_db")
COSMOS_CONTAINER = get_config("cosmos", "container", "COSMOS_CONTAINER", "rag_chunks")

ST_MODEL_NAME  = get_config("rag", "st_model", "ST_MODEL", "all-MiniLM-L6-v2")
COLLECTION     = get_config("rag", "collection", "COSMOS_COLLECTION", "rag_docs")
CHUNK_SIZE     = get_config_int("rag", "chunk_size", "CHUNK_SIZE", 500)
CHUNK_OVERLAP  = get_config_int("rag", "chunk_overlap", "CHUNK_OVERLAP", 50)
TOP_K          = get_config_int("rag", "top_k", "TOP_K", 3)
UPLOAD_DIR     = get_config("rag", "upload_dir", "UPLOAD_DIR", "./uploads")
PRINT_EMBEDDINGS      = get_config_bool("debug", "print_embeddings", "PRINT_EMBEDDINGS", True)
EMBEDDING_PRINT_LIMIT = get_config_int("debug", "embedding_print_limit", "EMBEDDING_PRINT_LIMIT", 3)
EMBEDDING_PRINT_DIMS  = get_config_int("debug", "embedding_print_dims", "EMBEDDING_PRINT_DIMS", 12)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None          # overrides TOP_K if provided
    collection: Optional[str] = None     # overrides COLLECTION if provided


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: list[str]


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int
    collection: str


class DeleteResponse(BaseModel):
    deleted: int
    collection: str


class CollectionStatsResponse(BaseModel):
    collection: str
    total_chunks: int


# ---------------------------------------------------------------------------
# Globals — initialised in lifespan
# ---------------------------------------------------------------------------
embed_model:   SentenceTransformer = None
openai_client: AzureOpenAI         = None
cosmos_container                  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup, release at shutdown."""
    global embed_model, openai_client, cosmos_container

    print(f"Loading sentence-transformer: {ST_MODEL_NAME}")
    embed_model = SentenceTransformer(ST_MODEL_NAME)

    print("Connecting to Azure OpenAI...")
    openai_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        timeout=AZURE_OPENAI_TIMEOUT,
    )

    print(f"Connecting to Azure Cosmos DB: {COSMOS_DATABASE}/{COSMOS_CONTAINER}")
    cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    database = cosmos_client.create_database_if_not_exists(id=COSMOS_DATABASE)
    cosmos_container = database.create_container_if_not_exists(
        id=COSMOS_CONTAINER,
        partition_key=PartitionKey(path="/collection"),
    )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print("Startup complete.")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG API using Azure OpenAI and Cosmos DB",
    description="FastAPI + Azure Cosmos DB + SentenceTransformers + Azure OpenAI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collection_name(name: Optional[str] = None) -> str:
    """Return the logical Cosmos DB collection name used as the partition key."""
    return name or COLLECTION


def embed(texts: list[str]) -> list[list[float]]:
    """Batch embed with sentence-transformers (normalised for cosine search)."""
    vectors = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def print_embedding_preview(filename: str, chunks: list[str], vectors: list[list[float]]) -> None:
    """Print a compact embedding preview for uploaded text files."""
    if not PRINT_EMBEDDINGS:
        return

    print(f"\nEmbedding preview for {filename}")
    print(f"Total chunks embedded: {len(vectors)}")

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        if i >= EMBEDDING_PRINT_LIMIT:
            remaining = len(vectors) - EMBEDDING_PRINT_LIMIT
            if remaining > 0:
                print(f"... skipped {remaining} more chunk embedding(s)")
            break

        preview = vector[:EMBEDDING_PRINT_DIMS]
        chunk_text = chunk.replace("\n", " ")[:120]
        print(f"Chunk {i}: {chunk_text!r}")
        print(f"Vector dimension: {len(vector)}")
        print(f"Vector preview: {preview}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity for two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]   # drop blank tail chunks


def count_chunks(col_name: str) -> int:
    query = "SELECT VALUE COUNT(1) FROM c WHERE c.collection = @collection"
    params = [{"name": "@collection", "value": col_name}]
    result = list(
        cosmos_container.query_items(
            query=query,
            parameters=params,
            partition_key=col_name,
        )
    )
    return result[0] if result else 0


def find_relevant_chunks(col_name: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    query = (
        "SELECT c.id, c.source, c.chunk_index, c.text, c.embedding "
        "FROM c WHERE c.collection = @collection"
    )
    params = [{"name": "@collection", "value": col_name}]
    items = cosmos_container.query_items(
        query=query,
        parameters=params,
        partition_key=col_name,
    )

    scored = []
    for item in items:
        score = cosine_similarity(query_vector, item["embedding"])
        scored.append({**item, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def build_context(matches: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract chunk text and source filenames from Cosmos DB search results."""
    sources = [item.get("source", "unknown") for item in matches]
    context = [
        f"[Source: {item.get('source', 'unknown')}]\n{item['text']}"
        for item in matches
    ]
    return context, sources


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# -- Health ------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model": ST_MODEL_NAME}


# -- Ingest ------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest(
    file: UploadFile = File(...),
    collection: Optional[str] = Query(None, description="Cosmos DB logical collection name"),
):
    """
    Upload a .txt file and ingest it into Cosmos DB.
    Each file is chunked, embedded locally, and stored with metadata.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported. Convert PDFs first.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with open(save_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(422, "File appears to be empty after chunking.")

    vectors = embed(chunks)
    print_embedding_preview(file.filename, chunks, vectors)
    col_name = collection_name(collection)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        cosmos_container.upsert_item(
            {
                "id": str(uuid.uuid4()),
                "collection": col_name,
                "source": file.filename,
                "chunk_index": i,
                "text": chunk,
                "embedding": vector,
            }
        )

    return IngestResponse(
        filename=file.filename,
        chunks_ingested=len(chunks),
        collection=col_name,
    )


# -- Ask ---------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(body: AskRequest):
    """
    Ask a question. Retrieves relevant chunks from Cosmos DB,
    then generates an answer via Azure OpenAI.
    """
    col_name = collection_name(body.collection)
    top_k = body.top_k or TOP_K
    print(f"Ask request: collection={col_name!r}, top_k={top_k}, query={body.query!r}")

    try:
        total_chunks = count_chunks(col_name)
    except Exception as exc:
        print(f"Cosmos DB count failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Cosmos DB count failed: {type(exc).__name__}: {exc}",
        ) from exc

    if total_chunks == 0:
        raise HTTPException(404, "Collection is empty — ingest a document first.")

    try:
        query_vector = embed([body.query])[0]
    except Exception as exc:
        print(f"Query embedding failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Query embedding failed: {type(exc).__name__}: {exc}",
        ) from exc

    try:
        matches = find_relevant_chunks(
            col_name=col_name,
            query_vector=query_vector,
            top_k=min(top_k, total_chunks),
        )
    except Exception as exc:
        print(f"Cosmos DB vector search failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Cosmos DB vector search failed: {type(exc).__name__}: {exc}",
        ) from exc

    context_parts, sources = build_context(matches)
    context = "\n\n---\n\n".join(context_parts)

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOY,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer using ONLY the context below. "
                        "If the answer is not in the context, say "
                        "'I don't know based on the provided documents.'\n\n"
                        f"CONTEXT:\n{context}"
                    ),
                },
                {"role": "user", "content": body.query},
            ],
            max_completion_tokens=1600,
        )
    except Exception as exc:
        print(f"Azure OpenAI completion failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Azure OpenAI completion failed: {type(exc).__name__}: {exc}",
        ) from exc

    return AskResponse(
        answer=response.choices[0].message.content,
        sources=list(set(sources)),
        chunks_used=[item["text"] for item in matches],
    )


# -- Collections -------------------------------------------------------------
@app.get("/collections", tags=["documents"])
def list_collections():
    """List all logical Cosmos DB collections."""
    query = "SELECT DISTINCT VALUE c.collection FROM c"
    collections = list(
        cosmos_container.query_items(
            query=query,
            enable_cross_partition_query=True,
        )
    )
    return {"collections": collections}


@app.get("/collections/{name}/stats", response_model=CollectionStatsResponse, tags=["documents"])
def collection_stats(name: str):
    """Get chunk count for a specific collection."""
    total = count_chunks(name)
    if total == 0:
        raise HTTPException(404, f"Collection '{name}' not found.")
    return CollectionStatsResponse(collection=name, total_chunks=total)


# -- Delete ------------------------------------------------------------------
@app.delete("/collections/{name}", response_model=DeleteResponse, tags=["documents"])
def delete_collection(name: str):
    """Delete an entire collection."""
    query = "SELECT c.id FROM c WHERE c.collection = @collection"
    params = [{"name": "@collection", "value": name}]
    items = list(
        cosmos_container.query_items(
            query=query,
            parameters=params,
            partition_key=name,
        )
    )
    if not items:
        raise HTTPException(404, f"Collection '{name}' not found.")

    for item in items:
        cosmos_container.delete_item(item=item["id"], partition_key=name)

    return DeleteResponse(deleted=len(items), collection=name)


@app.delete("/documents", response_model=DeleteResponse, tags=["documents"])
def delete_by_source(
    source: str = Query(..., description="Filename to delete, e.g. report.txt"),
    collection: Optional[str] = Query(None),
):
    """Delete all chunks that came from a specific file."""
    col_name = collection_name(collection)
    query = (
        "SELECT c.id FROM c "
        "WHERE c.collection = @collection AND c.source = @source"
    )
    params = [
        {"name": "@collection", "value": col_name},
        {"name": "@source", "value": source},
    ]
    items = list(
        cosmos_container.query_items(
            query=query,
            parameters=params,
            partition_key=col_name,
        )
    )
    if not items:
        raise HTTPException(404, f"No chunks found for source '{source}'.")

    for item in items:
        cosmos_container.delete_item(item=item["id"], partition_key=col_name)

    return DeleteResponse(deleted=len(items), collection=col_name)
