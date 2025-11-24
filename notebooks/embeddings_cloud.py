#!/usr/bin/env python
# upload_to_qdrant_cloud.py
#
# Migrate your YANG chunks + sensor catalog embeddings
# from local files to Qdrant Cloud.

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from openai import OpenAI

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Qdrant Cloud endpoint (the one you gave me)
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://55401e50-a35b-42ce-8759-08c6739076cc.eu-west-2-0.aws.cloud.qdrant.io",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # set in your shell

# Collections (same names as in your app)
FIXED_WINDOW_COLLECTION = "fixed_window_embeddings"
CATALOG_COLLECTION = "catalog_embeddings"

# Data paths (adjust if needed)
YANG_ROOT = "../data/yang/vendor/cisco/xr/701"
CATALOG_PATH = "../data/sensor_catalog.jsonl"

# Embedding model details
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536  # text-embedding-3-small → 1536 dims


# ---------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------

client_oa = OpenAI()  # uses OPENAI_API_KEY

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    resp = client_oa.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


@dataclass
class Chunk:
    id: int
    file_path: str
    chunk_index: int
    text: str


def load_yang_files(root: str) -> List[str]:
    pattern = os.path.join(root, "**", "*.yang")
    files = glob.glob(pattern, recursive=True)
    return files


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def build_chunks(root: str, max_chunks: Optional[int] = None) -> List[Chunk]:
    chunks: List[Chunk] = []
    files = load_yang_files(root)
    cid = 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        pieces = chunk_text(content)
        for i, piece in enumerate(pieces):
            if max_chunks is not None and len(chunks) >= max_chunks:
                print(f"Stopping early: reached {max_chunks} chunks")
                return chunks

            text = f"FILE: {os.path.basename(f)}\nCHUNK: {i}\n{piece}"
            chunks.append(Chunk(id=cid, file_path=f, chunk_index=i, text=text))
            cid += 1

    print(f"Loaded {len(files)} files, created {len(chunks)} chunks.")
    return chunks


def load_catalog_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    print(f"Loaded {len(rows)} catalog rows from {path}")
    return rows


def ensure_collection(name: str, dim: int = DIMENSION) -> None:
    if not qdrant.collection_exists(name):
        print(f"[+] Creating collection '{name}' in Qdrant Cloud...")
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"[=] Collection '{name}' already exists.")


# ---------------------------------------------------------------------
# UPLOAD FUNCTIONS
# ---------------------------------------------------------------------

def upload_fixed_window_embeddings(max_chunks: Optional[int] = None, batch_size: int = 64) -> None:
    """
    Upload YANG file chunks to Qdrant Cloud into FIXED_WINDOW_COLLECTION.
    """
    ensure_collection(FIXED_WINDOW_COLLECTION)

    chunks = build_chunks(YANG_ROOT, max_chunks=max_chunks)
    print(f"[+] Embedding and uploading {len(chunks)} YANG chunks...")

    batch: List[PointStruct] = []
    for chunk in chunks:
        vector = get_embedding(chunk.text)

        batch.append(
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "module": os.path.basename(chunk.file_path),
                    "file_path": chunk.file_path,
                    "chunk_index": chunk.chunk_index,
                    "text_preview": (
                        chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
                    ),
                    "source": "raw_yang_module",
                },
            )
        )

        if len(batch) >= batch_size:
            qdrant.upsert(
                collection_name=FIXED_WINDOW_COLLECTION,
                points=batch,
            )
            print(f"  Uploaded batch of {len(batch)} points.")
            batch = []

    # Remaining points
    if batch:
        qdrant.upsert(
            collection_name=FIXED_WINDOW_COLLECTION,
            points=batch,
        )
        print(f"  Uploaded final batch of {len(batch)} points.")

    print("[✓] Done uploading YANG chunks to fixed_window_embeddings.")


def upload_catalog_embeddings(limit: Optional[int] = None, batch_size: int = 64) -> None:
    """
    Upload sensor catalog rows to Qdrant Cloud into CATALOG_COLLECTION.
    """
    ensure_collection(CATALOG_COLLECTION)

    rows = load_catalog_rows(CATALOG_PATH, limit=limit)
    print(f"[+] Embedding and uploading {len(rows)} catalog rows...")

    batch: List[PointStruct] = []
    for idx, row in enumerate(rows):
        vector = get_embedding(row["search_text"])

        batch.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "yang_id": row["id"],
                    "module": row["module"],
                    "path": row["path"],
                    "protocol_tag": row["protocol_tag"],
                    "category": row["category"],
                    "kind": row["kind"],
                    "leaf_count": row["leaf_count"],
                    "description": row["description"],
                    "leaf_names": row["leaf_names"],
                },
            )
        )

        if len(batch) >= batch_size:
            qdrant.upsert(
                collection_name=CATALOG_COLLECTION,
                points=batch,
            )
            print(f"  Uploaded batch of {len(batch)} points.")
            batch = []

    if batch:
        qdrant.upsert(
            collection_name=CATALOG_COLLECTION,
            points=batch,
        )
        print(f"  Uploaded final batch of {len(batch)} points.")

    print("[✓] Done uploading catalog rows to catalog_embeddings.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload YANG chunks and sensor catalog embeddings to Qdrant Cloud."
    )
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Upload YANG chunks to fixed_window_embeddings.",
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Upload sensor catalog to catalog_embeddings.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of YANG chunks (for testing).",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Limit number of catalog rows (for testing).",
    )

    args = parser.parse_args()

    if not args.chunks and not args.catalog:
        print("Nothing to do. Use --chunks and/or --catalog.")
        return

    if args.chunks:
        upload_fixed_window_embeddings(max_chunks=args.max_chunks)

    if args.catalog:
        upload_catalog_embeddings(limit=args.limit_rows)


if __name__ == "__main__":
    main()
