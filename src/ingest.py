"""
Stage 1: Ingestion
===================
Parse -> Deduplicate -> Chunk -> Attach metadata.

Design choices (read this before you touch the config):

1. CHUNKING STRATEGY: recursive structural chunking, not semantic chunking.
   We split on paragraph breaks first, then sentences, then words, only
   falling through to a coarser separator when a piece is still too big.
   This respects the document's own structure (a contract clause, an HR
   policy section) instead of cutting mid-thought at a fixed character
   count. Semantic chunking (embedding every sentence and cutting at
   similarity valleys) is implemented separately in `semantic_chunk()`
   below as an OPTIONAL alternative you can A/B test later -- it costs an
   embedding call per sentence during ingestion, which is real money and
   latency at scale, so it should be a deliberate choice, not a default.

2. DEDUPLICATION happens at two levels:
   - Document-level: identical files (e.g. someone re-uploads the same
     contract) are caught by hashing the full normalized text.
   - Chunk-level: identical chunks across DIFFERENT documents (e.g. a
     boilerplate legal disclaimer repeated in every contract) are caught
     by hashing each chunk. This matters because near-duplicate chunks in
     your vector store don't just waste storage -- they can dominate the
     top-k results at query time and crowd out genuinely different
     relevant chunks. That's what the template means by "retrieval
     pollution."

3. METADATA is attached per chunk, not per document, because retrieval
   filters and citations operate at chunk granularity.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str          # stable id: doc_id + chunk index
    doc_id: str
    text: str
    chunk_index: int
    tenant_id: str
    access_level: str      # e.g. "internal", "public"
    version: int
    timestamp: str          # ISO 8601
    source_url: str
    start_char: Optional[int] = None   # position of this chunk in the ORIGINAL document text
    end_char: Optional[int] = None      # -- lets a citation point at an exact spot, not just "this file"
    metadata: dict = field(default_factory=dict)  # room for extras


def _normalize(text: str) -> str:
    """Collapse whitespace so trivial formatting differences don't defeat hashing."""
    return re.sub(r"\s+", " ", text).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def parse_document(path: Path) -> str:
    """
    Parse a source file into raw text.
    Only .txt/.md are handled inline here to keep this runnable with zero
    extra dependencies. PDFs are the common real-world case -- see the
    commented block below for how you'd plug in `pypdf` without changing
    anything downstream (parse_document always returns a plain string,
    that's the contract the rest of the pipeline relies on).
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    # if suffix == ".pdf":
    #     from pypdf import PdfReader
    #     reader = PdfReader(str(path))
    #     return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"No parser registered for {suffix}. Add one above.")


def recursive_chunk(
    text: str,
    max_chars: int = 500,
    overlap: int = 80,
) -> list[str]:
    """
    Structural recursive chunking.

    Strategy: try to split on the coarsest separator first (paragraph
    breaks). If a resulting piece is still bigger than max_chars, recurse
    into it with the next-finer separator (sentences, then words). This
    means a short, clean paragraph stays intact as ONE chunk even if
    there's room to pad it -- we don't force chunks to hit max_chars, we
    just cap them there. Overlap is applied only when we actually have to
    split a long piece, to preserve context across the cut.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(piece: str, seps: list[str]) -> list[str]:
        piece = piece.strip()
        if not piece:
            return []
        if len(piece) <= max_chars:
            return [piece]
        if not seps:
            # Nothing left to split on -- hard cut with overlap.
            out = []
            start = 0
            while start < len(piece):
                end = start + max_chars
                out.append(piece[start:end])
                start = end - overlap if end - overlap > start else end
            return out

        sep, rest_seps = seps[0], seps[1:]
        parts = [p for p in piece.split(sep) if p.strip()]
        if len(parts) == 1:
            # This separator didn't help (no split points found); go finer.
            return split(piece, rest_seps)

        chunks, buf = [], ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                if len(part) > max_chars:
                    chunks.extend(split(part, rest_seps))
                    buf = ""
                else:
                    buf = part
        if buf:
            chunks.append(buf)
        return chunks

    return split(text, separators)


def semantic_chunk_stub(text: str, embed_fn, similarity_threshold: float = 0.75) -> list[str]:
    """
    OPTIONAL alternative to recursive_chunk(). Not run by default -- wire
    it up in Stage 2 once you have an embedding function, then compare its
    retrieval quality against recursive_chunk() on the same corpus.

    Approach: split into sentences, embed each one, and start a new chunk
    whenever cosine similarity to the running chunk's centroid drops below
    `similarity_threshold` (i.e. the topic shifted).
    """
    import numpy as np

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return []

    embeddings = embed_fn(sentences)  # you supply this: list[str] -> np.ndarray [n, d]
    chunks, current, current_vecs = [], [sentences[0]], [embeddings[0]]

    for sent, vec in zip(sentences[1:], embeddings[1:]):
        centroid = np.mean(current_vecs, axis=0)
        sim = np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec) + 1e-9)
        if sim >= similarity_threshold:
            current.append(sent)
            current_vecs.append(vec)
        else:
            chunks.append(" ".join(current))
            current, current_vecs = [sent], [vec]
    chunks.append(" ".join(current))
    return chunks


def ingest_folder(
    folder: Path,
    tenant_id: str,
    access_level: str = "internal",
    version: int = 1,
    max_chars: int = 500,
    overlap: int = 80,
) -> tuple[list[Chunk], dict]:
    """
    Run the full Stage 1 pipeline over every .txt/.md file in `folder`.
    Returns (chunks, stats) where stats reports what dedup actually caught
    -- you want this visible, not silent, so you can sanity-check it.
    """
    seen_doc_hashes: set[str] = set()
    seen_chunk_hashes: set[str] = set()
    chunks: list[Chunk] = []
    stats = {"files_seen": 0, "docs_skipped_duplicate": 0, "chunks_skipped_duplicate": 0, "chunks_kept": 0}

    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        stats["files_seen"] += 1

        raw_text = parse_document(path)
        doc_hash = _hash(raw_text)
        if doc_hash in seen_doc_hashes:
            stats["docs_skipped_duplicate"] += 1
            continue
        seen_doc_hashes.add(doc_hash)

        doc_id = doc_hash[:12]  # content-derived id -- re-ingesting the same content is a no-op, not a new doc_id
        pieces = recursive_chunk(raw_text, max_chars=max_chars, overlap=overlap)

        # Locate each chunk's exact position in the original text. We search
        # forward from the end of the PREVIOUS match (not from 0 each time),
        # both for speed and so that if the same short phrase legitimately
        # appears twice in a document, each occurrence still gets matched to
        # the right chunk in reading order rather than both matching the
        # first occurrence. This is a heuristic (str.find on the stripped
        # piece) -- it can't perfectly handle a document where two DIFFERENT
        # chunks happen to contain byte-for-byte identical text, but that's
        # rare and the fallback (None, None) below makes the gap visible
        # instead of silently wrong.
        search_from = 0
        offsets: list[tuple[Optional[int], Optional[int]]] = []
        for piece in pieces:
            needle = piece.strip()
            pos = raw_text.find(needle, search_from)
            if pos == -1:
                offsets.append((None, None))
            else:
                offsets.append((pos, pos + len(needle)))
                search_from = pos + 1  # allow future overlap, still monotonic

        for idx, (piece, (start_char, end_char)) in enumerate(zip(pieces, offsets)):
            chunk_hash = _hash(piece)
            if chunk_hash in seen_chunk_hashes:
                stats["chunks_skipped_duplicate"] += 1
                continue
            seen_chunk_hashes.add(chunk_hash)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{idx}",
                    doc_id=doc_id,
                    text=piece,
                    chunk_index=idx,
                    tenant_id=tenant_id,
                    access_level=access_level,
                    version=version,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source_url=str(path),
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            stats["chunks_kept"] += 1

    return chunks, stats


if __name__ == "__main__":
    # Quick self-test with the sample docs -- run this file directly.
    folder = Path(__file__).parent.parent / "data" / "sample_docs"
    chunks, stats = ingest_folder(folder, tenant_id="tenant_demo")

    print("=== Ingestion stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\n=== {len(chunks)} chunks produced ===")
    for c in chunks:
        print(f"\n[{c.chunk_id}] doc={c.doc_id} tenant={c.tenant_id} idx={c.chunk_index} len={len(c.text)}")
        print(f"  {c.text[:120]}{'...' if len(c.text) > 120 else ''}")