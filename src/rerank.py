"""
Stage 4: Reranking
====================
Re-score retrieval's candidates with a cross-encoder before anything goes
to the LLM. See module docstring context in the chat -- short version:
bi-encoder (Stage 3) is fast+broad, cross-encoder (this stage) is
slow+accurate, and only ever runs on the small candidate set, never the
full corpus.
"""

import os
from dataclasses import dataclass

from retrieval import RetrievedChunk, _tokenize


@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    payload: dict
    rerank_score: float
    retrieval_rrf_score: float


def rerank_cohere(
    query: str, candidates: list[RetrievedChunk], top_n: int = 5, model: str = "rerank-english-v3.0"
) -> list[RerankedChunk]:
    """REAL reranking. Requires COHERE_API_KEY in your environment."""
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    docs = [c.text for c in candidates]
    response = co.rerank(query=query, documents=docs, top_n=top_n, model=model)

    return [
        RerankedChunk(
            chunk_id=candidates[r.index].chunk_id,
            text=candidates[r.index].text,
            payload=candidates[r.index].payload,
            rerank_score=r.relevance_score,
            retrieval_rrf_score=candidates[r.index].rrf_score,
        )
        for r in response.results
    ]


def rerank_mock(query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RerankedChunk]:
    """
    FAKE reranker for testing the code path without a Cohere key. Scores by
    raw word overlap between query and chunk -- crude, NOT a real
    cross-encoder (it can't judge meaning, negation, or word order at all,
    it's just counting shared words). Only here to prove the wiring works.
    """
    query_tokens = set(_tokenize(query))
    scored = []
    for c in candidates:
        chunk_tokens = set(_tokenize(c.text))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        scored.append((c, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        RerankedChunk(
            chunk_id=c.chunk_id, text=c.text, payload=c.payload,
            rerank_score=score, retrieval_rrf_score=c.rrf_score,
        )
        for c, score in scored[:top_n]
    ]


if __name__ == "__main__":
    from retrieval import HybridRetriever

    retriever = HybridRetriever()
    retriever.build_bm25_index()

    query = "What are the payment terms for ACME-2463?"
    candidates = retriever.hybrid_search(query, top_k=10, tenant_id="tenant_demo")

    print(f"Retrieved {len(candidates)} candidates before reranking:")
    for c in candidates:
        print(f"  chunk_id={c.chunk_id}  rrf={c.rrf_score:.5f}")

    if os.environ.get("COHERE_API_KEY"):
        print("\nCOHERE_API_KEY found -- using real Cohere reranker.\n")
        reranked = rerank_cohere(query, candidates, top_n=3)
    else:
        print("\nNo COHERE_API_KEY -- using MOCK reranker (word overlap only, not a real cross-encoder).\n")
        reranked = rerank_mock(query, candidates, top_n=3)

    print(f"Top {len(reranked)} after reranking:")
    for r in reranked:
        print(f"  chunk_id={r.chunk_id}  rerank_score={r.rerank_score:.4f}  (was rrf={r.retrieval_rrf_score:.5f})")
        print(f"    {r.text[:90]!r}...")

    retriever.client.close()