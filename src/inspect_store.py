"""
Diagnostic script -- NOT part of the pipeline, just a way to look inside
Qdrant after you've already run embed_store.py at least once.

    python src/inspect_store.py
"""

from embed_store import COLLECTION_NAME, get_client

client = get_client()  # points at the same ./qdrant_data folder embed_store.py wrote to

# Grab a handful of points without needing to know any IDs in advance.
# scroll() is Qdrant's "just give me some points" call, as opposed to
# query_points() which needs a query vector to search against.
points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_vectors=True,   # False by default -- vectors are the expensive part to return
    with_payload=True,
)

for p in points:
    print(f"id: {p.id}")
    print(f"chunk_id: {p.payload['chunk_id']}")
    print(f"text: {p.payload['text'][:60]!r}...")
    print(f"vector (first 10 of {len(p.vector)} numbers): {p.vector[:10]}")
    print()

client.close()