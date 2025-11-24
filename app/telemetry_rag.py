#!/usr/bin/env python
# telemetry_rag.py
# Simple RAG pipeline for Cisco IOS XR telemetry config generation using Qdrant + OpenAI

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "fixed_window_embeddings")

XR_TELEMETRY_SYSTEM_PROMPT = """
You are a Cisco IOS XR network engineer.

Always output telemetry model-driven configuration for IOS XR 7.x
using EXACTLY this structure (adapt names as needed):

telemetry model-driven
 sensor-group <SENSOR_GROUP_NAME>
  sensor-path <PATH_1>
  sensor-path <PATH_2>
 !
 destination-group DG-GRPC
  address-family ipv4
   destination <DEST_IP>
    port <DEST_PORT>
    encoding self-describing-gpb
    protocol grpc no-tls
 !
 subscription <SUBSCRIPTION_NAME>
  sensor-group-id <SENSOR_GROUP_NAME> sample-interval <INTERVAL_MS>
  destination-group-id DG-GRPC
 !

Rules:
- Do NOT use 'stream', 'transport', 'destination-ip', 'destination-port', or 'no tls' commands.
- Put ALL sensor-path lines inside a sensor-group.
- Put ALL destination settings inside a destination-group as shown.
- Use only IOS XR syntax.
- Use the CONTEXT below only to choose relevant sensor-paths.
- Output only configuration, no explanations.
""".strip()

# ---------------------------------------------------------------------
# Global clients (lazy singletons)
# ---------------------------------------------------------------------

_client_oa: Optional[OpenAI] = None
_client_qdrant: Optional[QdrantClient] = None


def get_openai_client() -> OpenAI:
    """Return a singleton OpenAI client (uses OPENAI_API_KEY env var)."""
    global _client_oa
    if _client_oa is None:
        _client_oa = OpenAI()
    return _client_oa


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client_qdrant
    if _client_qdrant is None:
        _client_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client_qdrant


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Chunk:
    id: int
    file_path: str
    chunk_index: int
    text: str
    score: float = 0.0


# ---------------------------------------------------------------------
# Embeddings & Retrieval
# ---------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Compute an embedding for the given text using OpenAI.
    """
    client = get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def retrieve_chunks(
    query: str,
    collection_name: str,
    top_k: int = 8,
    query_filter: Optional[Filter] = None,
    score_threshold: Optional[float] = None,
) -> List[Chunk]:
    """
    Universal retrieval function — works with ANY Qdrant collection.

    Args:
        query: Natural language query.
        collection_name: Name of the Qdrant collection (e.g. "yang_sensors", "catalog_embeddings").
        top_k: Max number of results to return.
        query_filter: Optional Qdrant Filter (e.g. by protocol_tag, vendor, etc.).
        score_threshold: If provided, discard hits below this score.

    Returns:
        List[Chunk]: RAG chunks with text + metadata.
    """
    client = get_qdrant_client()
    query_vector = get_embedding(query)

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    hits = result.points
    chunks: List[Chunk] = []

    for hit in hits:
        if score_threshold is not None and hit.score < score_threshold:
            continue

        payload: Dict[str, Any] = hit.payload or {}

        # Different collections may store text in different keys
        base_text = (
            payload.get("text")
            or payload.get("text_preview")
            or (
                (payload.get("description", "") or "")
                + "\n"
                + (payload.get("path", "") or "")
            )
        )

        chunk = Chunk(
            id=hit.id,
            file_path=payload.get("file_path") or payload.get("module", "unknown"),
            chunk_index=payload.get("chunk_index", 0),
            text=base_text or "",
            score=hit.score,  # type: ignore[arg-type]
        )
        chunks.append(chunk)

    # Sort by score descending and enforce final top_k after thresholding
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_k]


# ---------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------

def build_context_text(retrieved: List[Chunk]) -> str:
    """
    Turn retrieved chunks into a single context string for the LLM.
    """
    parts: List[str] = []
    for c in retrieved:
        parts.append(
            f"---\n"
            f"Source file: {os.path.basename(c.file_path)} | chunk {c.chunk_index}\n"
            f"{c.text}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------
# LLM call for telemetry config
# ---------------------------------------------------------------------

def generate_telemetry_config(
    user_query: str,
    top_k: int = 8,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    system_prompt: str = XR_TELEMETRY_SYSTEM_PROMPT,
    query_filter: Optional[Filter] = None,
    score_threshold: Optional[float] = None,
    temperature: float = 0.2,
) -> Tuple[str, List[Chunk]]:
    """
    High-level RAG pipeline: query → retrieve chunks → build context → ask LLM.

    Args:
        user_query: Natural language request, e.g.
            "Generate telemetry configuration for Cisco IOS XR about BGP..."
        top_k: Number of chunks to retrieve.
        collection_name: Qdrant collection to use.
        system_prompt: System prompt to control IOS XR syntax & structure.
        query_filter: Optional Qdrant filter.
        score_threshold: Optional score cutoff for retrieved points.
        temperature: LLM sampling temperature.

    Returns:
        (config_text, retrieved_chunks)
    """
    retrieved = retrieve_chunks(
        query=user_query,
        collection_name=collection_name,
        top_k=top_k,
        query_filter=query_filter,
        score_threshold=score_threshold,
    )

    context = build_context_text(retrieved)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"USER REQUEST:\n{user_query}\n\n"
                "Return only the telemetry model-driven configuration."
            ),
        },
    ]

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )

    content = resp.choices[0].message.content if resp.choices else ""
    config_text = (content or "").strip()
    return config_text, retrieved


# ---------------------------------------------------------------------
# JSON-friendly wrapper (very MCP-friendly)
# ---------------------------------------------------------------------

def run_rag_telemetry_query(
    user_query: str,
    top_k: int = 8,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    system_prompt: str = XR_TELEMETRY_SYSTEM_PROMPT,
    query_filter: Optional[Filter] = None,
    score_threshold: Optional[float] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns everything as a JSON-serializable dict.

    This is a perfect function to expose as an MCP tool later.

    Returns:
        {
          "config": "<IOS XR telemetry config>",
          "retrieved_chunks": [
            {
              "id": ...,
              "file_path": ...,
              "chunk_index": ...,
              "text": ...,
              "score": ...
            }, ...
          ]
        }
    """
    config_text, retrieved_chunks = generate_telemetry_config(
        user_query=user_query,
        top_k=top_k,
        collection_name=collection_name,
        system_prompt=system_prompt,
        query_filter=query_filter,
        score_threshold=score_threshold,
        temperature=temperature,
    )

    return {
        "config": config_text,
        "retrieved_chunks": [asdict(c) for c in retrieved_chunks],
    }


# ---------------------------------------------------------------------
# CLI entrypoint (for quick local tests)
# ---------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple RAG for Cisco IOS XR telemetry configuration using Qdrant + OpenAI."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "Generate telemetry configuration for Cisco IOS XR about BGP. "
            "Use gRPC with no TLS, telemetry server 192.0.2.0 port 57500. "
            "Choose relevant BGP sensor paths."
        ),
        help="Natural language query for telemetry configuration.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON (config + retrieved chunks) instead of plain config.",
    )

    args = parser.parse_args()

    if args.json:
        result = run_rag_telemetry_query(
            user_query=args.query,
            top_k=args.top_k,
            collection_name=args.collection,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        config_text, _ = generate_telemetry_config(
            user_query=args.query,
            top_k=args.top_k,
            collection_name=args.collection,
        )
        print(config_text)


if __name__ == "__main__":
    main()
