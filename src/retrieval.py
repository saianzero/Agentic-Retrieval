"""
Stage 3: Hybrid Retrieval
==========================
BM25 (keyword) + dense (semantic) search, run independently, merged with
Reciprocal Rank Fusion (RRF).

Why both, in one line: BM25 answers "same words," dense answers "same
meaning." Your template's own example nails it -- "ACME-2463" is a BM25
problem (exact identifier), "30 days payment terms" is a dense problem
(intent, regardless of exact phrasing).

Why RRF, not just averaging the two scores: BM25 scores and cosine
similarity scores are on incompatible scales (BM25 is unbounded, cosine is
-1..1). Averaging them directly is meaningless. RRF ignores raw scores
entirely and uses only RANK POSITION -- robust, and needs zero score-scale
tuning.

    rrf_score(chunk) = sum, over every ranked list it appears in, of
                        1 / (rrf_k + rank_in_that_list)

rrf_k (60 is the standard default from the original paper) is a damping
constant: it stops a #1 finish in one list from completely dominating a
chunk that placed solidly (but not first) in both lists.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

from embed_store import COLLECTION_NAME, get_client, get_embeddings_mock, get_embeddings_openai


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    payload: dict
    rrf_score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumerics. Good enough for a demo;
    production BM25 often adds stemming (running/runs/ran -> run)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, client=None):
        self.client = client or get_client()
        self._bm25 = None
        self._chunk_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._payloads: dict[str, dict] = {}

    def build_bm25_index(self) -> None:
        """
        BM25 has no persistence of its own (unlike Qdrant) -- we rebuild an
        in-memory index from whatever's currently in Qdrant. Fine at this
        scale; at real scale you'd use a proper text search engine
        (Elasticsearch/OpenSearch, or Qdrant's own built-in sparse vectors)
        instead of rebuilding this on every process start.
        """
        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME, limit=10_000, with_payload=True, with_vectors=False
        )
        self._chunk_ids = [p.payload["chunk_id"] for p in points]
        self._texts = {p.payload["chunk_id"]: p.payload["text"] for p in points}
        self._payloads = {p.payload["chunk_id"]: p.payload for p in points}

        tokenized_corpus = [_tokenize(self._texts[cid]) for cid in self._chunk_ids]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def bm25_search(self, query: str, top_k: int = 10) -> list[str]:
        """Chunk_ids, best match first."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked[:top_k] if score > 0]

    def dense_search(self, query: str, top_k: int = 10, tenant_id: Optional[str] = None) -> list[str]:
        """
        Chunk_ids, best match first. Note WHERE the tenant filter is applied:
        inside query_points(), as part of the ANN search itself -- not as a
        post-hoc filter on the results. If we searched top_k=10 first and
        THEN filtered by tenant, a tenant with few docs could end up with
        zero results just because their chunks got pushed outside the top 10
        globally, even though they were the best match within their own
        tenant. Qdrant applies the filter during graph traversal, so this
        can't happen.
        """
        embed_fn = get_embeddings_openai if os.environ.get("OPENAI_API_KEY") else get_embeddings_mock
        query_vector = embed_fn([query])[0]

        query_filter = None
        if tenant_id:
            query_filter = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [r.payload["chunk_id"] for r in results]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidates_per_method: int = 10,
        tenant_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        bm25_ranked = self.bm25_search(query, top_k=candidates_per_method)
        dense_ranked = self.dense_search(query, top_k=candidates_per_method, tenant_id=tenant_id)

        scores: dict[str, float] = {}
        bm25_rank_map: dict[str, int] = {}
        dense_rank_map: dict[str, int] = {}

        for rank, cid in enumerate(bm25_ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            bm25_rank_map[cid] = rank + 1

        for rank, cid in enumerate(dense_ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            dense_rank_map[cid] = rank + 1

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                chunk_id=cid,
                text=self._texts[cid],
                payload=self._payloads[cid],
                rrf_score=scores[cid],
                bm25_rank=bm25_rank_map.get(cid),
                dense_rank=dense_rank_map.get(cid),
            )
            for cid in ranked_ids
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.build_bm25_index()
    print(f"BM25 index built over {len(retriever._chunk_ids)} chunks.\n")

    mode = "real" if os.environ.get("OPENAI_API_KEY") else "MOCK (no OPENAI_API_KEY -- dense ranking will be meaningless noise)"
    print(f"Dense search mode: {mode}\n")

    # --- Test 1: the exact-identifier case your template calls out ---
    query = "ACME-2463 payment terms"
    results = retriever.hybrid_search(query, top_k=5, tenant_id="tenant_demo")
    print(f"Query: {query!r}")
    for r in results:
        print(f"  chunk_id={r.chunk_id}  rrf={r.rrf_score:.5f}  bm25_rank={r.bm25_rank}  dense_rank={r.dense_rank}")
        print(f"    {r.text[:90]!r}...")

    # --- Test 2: prove the tenant filter is actually doing work, not decorative ---
    print("\nFiltering dense search for a tenant that doesn't exist ('tenant_ghost'):")
    ghost_results = retriever.dense_search(query, top_k=5, tenant_id="tenant_ghost")
    print(f"  results: {ghost_results}  (should be empty)")

    retriever.client.close()