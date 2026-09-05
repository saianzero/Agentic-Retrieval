"""
Stage 5: Generation + Citations
==================================
Pass reranked chunks to the LLM, get an answer, map it back to real
sources -- without trusting the LLM to output real chunk_ids/UUIDs itself.

Design: the model only ever sees short numbered labels ([1], [2], [3]) for
each chunk and is told to cite using those. WE built the prompt, so we
already know exactly which real chunk_id/source_url/text each label maps
to. Resolving [1] -> real metadata is a lookup in a dict we already have,
not something the model needs to get right. This is deliberately the
least the model could possibly be asked to do correctly.
"""

import os
import re
import time
from dataclasses import dataclass, field

from rerank import RerankedChunk


@dataclass
class Citation:
    label: int
    chunk_id: str
    source_url: str
    text: str
    start_char: int | None = None
    end_char: int | None = None

    def locator(self) -> str:
        """Human-readable pointer into the exact source location, when we have one."""
        if self.start_char is None:
            return f"{self.source_url} (exact position unknown)"
        return f"{self.source_url} [chars {self.start_char}-{self.end_char}]"


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    latency_seconds: dict = field(default_factory=dict)


SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided below.
Rules:
- If the passages don't contain the answer, say so plainly. Do not guess or use outside knowledge.
- Cite the passage(s) you used for each claim using its number in square brackets, e.g. [1] or [1][3].
- Keep the answer concise."""


def build_prompt(query: str, chunks: list[RerankedChunk]) -> tuple[str, dict[int, RerankedChunk]]:
    """Numbers the chunks 1..N and returns both the user-facing prompt text
    and the label->chunk lookup the backend will use afterward."""
    label_map: dict[int, RerankedChunk] = {}
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        label_map[i] = c
        context_blocks.append(f"[{i}] {c.text}")

    user_prompt = (
        "Context passages:\n\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {query}"
    )
    return user_prompt, label_map


def _resolve_citations(answer_text: str, label_map: dict[int, RerankedChunk]) -> list[Citation]:
    """Find every [n] the model actually used, in order of first appearance,
    and resolve each to the real chunk behind it. Labels the model
    hallucinates (out of range) are silently dropped, not fabricated into
    fake citations."""
    seen = []
    for n_str in re.findall(r"\[(\d+)\]", answer_text):
        n = int(n_str)
        if n in label_map and n not in seen:
            seen.append(n)

    return [
        Citation(
            label=n,
            chunk_id=label_map[n].chunk_id,
            source_url=label_map[n].payload["source_url"],
            text=label_map[n].text,
            start_char=label_map[n].payload.get("start_char"),
            end_char=label_map[n].payload.get("end_char"),
        )
        for n in seen
    ]


def generate_answer_openai(
    query: str, chunks: list[RerankedChunk], model: str = "gpt-4o-mini"
) -> GeneratedAnswer:
    """REAL generation via OpenAI. Requires OPENAI_API_KEY."""
    from openai import OpenAI

    client = OpenAI()
    user_prompt, label_map = build_prompt(query, chunks)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    latency = time.perf_counter() - t0

    answer_text = response.choices[0].message.content
    citations = _resolve_citations(answer_text, label_map)
    return GeneratedAnswer(answer=answer_text, citations=citations, latency_seconds={"generation": latency})


def generate_answer_mock(query: str, chunks: list[RerankedChunk]) -> GeneratedAnswer:
    """
    FAKE generation for testing the prompt-building and citation-resolving
    logic without an API call. Just echoes back the top chunk with a
    citation marker -- proves the label->real-chunk mapping works, proves
    NOTHING about answer quality (there's no real reasoning happening).
    """
    user_prompt, label_map = build_prompt(query, chunks)
    top_label = 1
    fake_answer = f"Based on the top passage: {chunks[0].text[:80]}... [{top_label}]"
    citations = _resolve_citations(fake_answer, label_map)
    return GeneratedAnswer(answer=fake_answer, citations=citations, latency_seconds={"generation": 0.0})


if __name__ == "__main__":
    from retrieval import HybridRetriever
    from rerank import rerank_cohere, rerank_mock

    t_start = time.perf_counter()
    retriever = HybridRetriever()
    retriever.build_bm25_index()

    query = "What are the payment terms for ACME-2463?"

    t0 = time.perf_counter()
    candidates = retriever.hybrid_search(query, top_k=10, tenant_id="tenant_demo")
    t_retrieval = time.perf_counter() - t0

    t0 = time.perf_counter()
    if os.environ.get("COHERE_API_KEY"):
        reranked = rerank_cohere(query, candidates, top_n=3)
    else:
        reranked = rerank_mock(query, candidates, top_n=3)
    t_rerank = time.perf_counter() - t0

    if os.environ.get("OPENAI_API_KEY"):
        print("Using real OpenAI generation.\n")
        result = generate_answer_openai(query, reranked)
    else:
        print("No OPENAI_API_KEY -- using MOCK generation (proves citation wiring, not answer quality).\n")
        result = generate_answer_mock(query, reranked)

    t_total = time.perf_counter() - t_start

    print(f"Query: {query}\n")
    print(f"Answer:\n{result.answer}\n")
    print("Citations (resolved by backend, not the model):")
    for c in result.citations:
        print(f"  [{c.label}] chunk_id={c.chunk_id}")
        print(f"      {c.locator()}")
        print(f"      {c.text[:80]!r}...")

    print("\nLatency breakdown:")
    print(f"  retrieval (BM25+dense+RRF): {t_retrieval*1000:.1f} ms")
    print(f"  rerank:                     {t_rerank*1000:.1f} ms")
    print(f"  generation:                 {result.latency_seconds.get('generation', 0)*1000:.1f} ms")
    print(f"  total pipeline:              {t_total*1000:.1f} ms")

    retriever.client.close()