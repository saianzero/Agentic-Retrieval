"""
Stage 2: Embeddings + Vector Storage
======================================
Turn Chunk objects into vectors, store them in Qdrant.

Design choices:

1. LOCAL QDRANT, NO DOCKER: QdrantClient(path=...) runs a real embedded
   Qdrant engine on disk, single-process. This is perfect for a demo/dev
   machine. For a real multi-service deployment you'd switch to
   QdrantClient(url="http://localhost:6333") pointing at a Qdrant server
   (Docker or cloud) -- same API, so the switch is a one-line change later.

2. STABLE IDS: each chunk gets a UUID derived from its chunk_id (not a
   random one, not an incrementing counter). This means re-running
   ingestion on the same content overwrites the same point instead of
   creating a duplicate -- ingestion becomes "idempotent" (safe to re-run).

3. HNSW CONFIG IS EXPLICIT, NOT DEFAULT: we set m and ef_construct by hand
   so they're visible knobs, not a black box. See create_collection() for
   what each one trades off.

4. PAYLOAD = METADATA + TEXT, STORED WITH THE VECTOR: Qdrant returns the
   chunk text and all metadata fields in the same call that returns the
   nearest vectors. You don't do a separate lookup after search.
"""

import hashlib
import os
import uuid
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

from ingest import Chunk

EMBEDDING_DIM = 1536  # matches OpenAI text-embedding-3-small
COLLECTION_NAME = "rag_demo"


# ---------------------------------------------------------------------------
# Embedding functions -- swap MOCK for REAL once you have OPENAI_API_KEY set
# ---------------------------------------------------------------------------

def get_embeddings_openai(
    texts: list[str], model: str = "text-embedding-3-small", batch_size: int = 100
) -> list[list[float]]:
    """
    REAL embeddings via OpenAI. Requires OPENAI_API_KEY in your environment.
    Batches requests (see module docstring for why).
    """
    from openai import OpenAI

    client = OpenAI()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def get_embeddings_mock(texts: list[str], dim: int = EMBEDDING_DIM) -> list[list[float]]:
    """
    FAKE embeddings, deterministic per input text (same text -> same vector,
    every run). This is ONLY here so we can test that storage/search/schema
    logic works with zero network access and zero cost. It carries NO real
    semantic meaning -- two texts about the same topic will NOT end up
    close together like they would with a real model. Do not use this for
    anything except plumbing tests.
    """
    vectors = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=dim)
        v = v / np.linalg.norm(v)  # normalize -> unit length, since we use cosine distance
        vectors.append(v.tolist())
    return vectors


# ---------------------------------------------------------------------------
# Qdrant storage
# ---------------------------------------------------------------------------

def get_client(path: str = "./qdrant_data") -> QdrantClient:
    """Embedded, local, no server process. Swap to url=... for a real deployment."""
    return QdrantClient(path=path)


def create_collection(client: QdrantClient, dim: int = EMBEDDING_DIM, recreate: bool = False) -> None:
    """
    This is where your template's "build an ANN index" step actually
    happens -- except we don't build it, we configure it:

      - m: how many graph connections each vector keeps to its nearest
        neighbors. Think of it like how many friends each point stays in
        touch with. Higher m = better recall, more memory, slower to build.
        Qdrant's default is 16 -- we set it explicitly so it's not hidden.

      - ef_construct: how hard Qdrant searches for good neighbors WHILE
        building the graph. Higher = better quality graph, slower ingestion.
        This only costs time at ingestion, not at query time.

    Neither of these matters at your current scale (a handful of chunks).
    They start mattering at hundreds of thousands+ vectors -- but the point
    of exposing them now is so you know where the dial is when it does.
    """
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )


def _stable_id(chunk_id: str) -> str:
    """Deterministic UUID from a chunk_id string -- makes re-ingestion idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=_stable_id(c.chunk_id),
            vector=emb,
            payload={
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "tenant_id": c.tenant_id,
                "access_level": c.access_level,
                "version": c.version,
                "timestamp": c.timestamp,
                "source_url": c.source_url,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            },
        )
        for c, emb in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    from pathlib import Path

    from ingest import ingest_folder

    # --- Step 1 pipeline output feeds directly into Step 2 ---
    folder = Path(__file__).parent.parent / "data" / "sample_docs"
    chunks, stats = ingest_folder(folder, tenant_id="tenant_demo")
    print(f"Ingested {len(chunks)} chunks from {stats['files_seen']} files "
          f"({stats['docs_skipped_duplicate']} duplicate doc(s) skipped)\n")

    # --- Embed: use real OpenAI if a key is set, otherwise fall back to mock.
    #     This env-based switch is itself a common pattern -- same code path
    #     for local dev/testing and real usage, no code change needed.
    texts = [c.text for c in chunks]
    if os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY found -- using real embeddings.\n")
        embeddings = get_embeddings_openai(texts)
    else:
        print("No OPENAI_API_KEY found -- using MOCK embeddings (plumbing test only, "
              "not real semantic search). Set the env var to use the real thing.\n")
        embeddings = get_embeddings_mock(texts)
    print(f"Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}\n")

    # --- Store ---
    client = get_client()
    create_collection(client, recreate=True)
    upsert_chunks(client, chunks, embeddings)
    count = client.count(COLLECTION_NAME).count
    print(f"Stored {count} points in Qdrant collection '{COLLECTION_NAME}'\n")

    # --- Prove search works mechanically (mock vectors carry no real meaning,
    #     so treat this as a plumbing test, not a relevance test) ---
    embed_fn = get_embeddings_openai if os.environ.get("OPENAI_API_KEY") else get_embeddings_mock
    query_text = "What are the payment terms?"
    query_vector = embed_fn([query_text])[0]
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=3
    ).points

    label = "real" if os.environ.get("OPENAI_API_KEY") else "MOCK -- ignore relevance, this only checks wiring"
    print(f"Query: {query_text!r}")
    print(f"Top 3 nearest ({label} embeddings):")
    for r in results:
        print(f"  score={r.score:.4f}  chunk_id={r.payload['chunk_id']}  "
              f"tenant={r.payload['tenant_id']}  text={r.payload['text'][:60]!r}...")

    client.close()  # explicit cleanup -- avoids a harmless-but-noisy warning at interpreter shutdown