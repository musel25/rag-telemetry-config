#!/usr/bin/env python
# coding: utf-8

# Init qdrant
# 
# `docker run -p 6333:6333 -p 6334:6334 \
#     -v $(pwd)/qdrant_storage:/qdrant/storage:z \
#     qdrant/qdrant:latest`

# In[27]:


from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


# In[28]:



# Local Qdrant example – adjust for Cloud (host, api_key, etc.)
client = QdrantClient(host="localhost", port=6333)


# In[29]:


# Cell 2: load YANG catalog
import json

CATALOG_PATH = "../data/sensor_catalog.jsonl"  # or .json

all_rows = []

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        all_rows.append(json.loads(line))

len(all_rows)


# In[30]:


from openai import OpenAI
from qdrant_client.models import PointStruct

client_oa = OpenAI()  # uses OPENAI_API_KEY from env

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536  # for text-embedding-3-small
def get_embedding(text: str) -> list[float]:
    resp = client_oa.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding

# In[31]:


from dataclasses import dataclass
from typing import List, Dict, Any

import os
import glob

# Cell 2: load & chunk YANG files

YANG_ROOT = "../data/yang/vendor/cisco/xr/701"  # adapt if your path is different

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
    # naive fixed-size char chunking
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def build_chunks(root: str, max_chunks: int = None) -> List[Chunk]:
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
                return chunks  # ← early return
            
            # add a tiny header so the model knows where this comes from
            text = f"FILE: {os.path.basename(f)}\nCHUNK: {i}\n{piece}"
            chunks.append(Chunk(id=cid, file_path=f, chunk_index=i, text=text))
            cid += 1
    print(f"Loaded {len(files)} files, created {len(chunks)} chunks.")
    return chunks

chunks = build_chunks(YANG_ROOT,max_chunks=10)


# In[ ]:


# Create new collection for raw YANG chunks
collection_name = "fixed_window_embeddings"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )

print(f"Created collection: {collection_name}")

# In[33]:


# Embed and upload only your 10 chunks

points = []
for chunk in chunks:  # ← only your 10 chunks
    vector = get_embedding(chunk.text)
    
    points.append(PointStruct(
        id=chunk.id,
        vector=vector,
        payload={
            "module": os.path.basename(chunk.file_path),
            "file_path": chunk.file_path,
            "chunk_index": chunk.chunk_index,
            "text_preview": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
            "source": "raw_yang_module"
        }
    ))

client.upsert(collection_name=collection_name, points=points)
print(f"Uploaded {len(points)} chunks to {collection_name}")

# In[34]:


# Example: Scroll all points in a collection (replace 'your_collection' with your actual collection name)
points = client.scroll(
    collection_name="fixed_window_embeddings",
    limit=10,  # Fetch first 10 points; use with_offset for pagination
    with_vectors=True,  # Include the actual vector values
    with_payload=True   # Include metadata/payloads
)

# Print the results
for point in points[0][:3]:
    print(f"Point ID: {point.id}")
    print(f"Vector values (first 5 dims): {point.vector[:5]}...")  # Vectors are lists of floats; slice to avoid spam
    print(f"Payload: {point.payload}")
    print("---")

# In[35]:


if not client.collection_exists("catalog_embeddings"):
    client.create_collection(
        collection_name="catalog_embeddings",
        vectors_config=VectorParams(size=1536, distance="Cosine"),
    )

# In[36]:


from qdrant_client.models import PointStruct

points = []

for idx, row in enumerate(all_rows[:10]):
    vector = get_embedding(row["search_text"])

    point = PointStruct(
        id=idx,  # ✅ valid: unsigned integer
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
    points.append(point)

client.upsert(
    collection_name="catalog_embeddings",
    points=points,
)


# In[37]:


# Example: Scroll all points in a collection (replace 'your_collection' with your actual collection name)
points = client.scroll(
    collection_name="catalog_embeddings",
    limit=10,  # Fetch first 10 points; use with_offset for pagination
    with_vectors=True,  # Include the actual vector values
    with_payload=True   # Include metadata/payloads
)

# Print the results
for point in points[0][:3]:
    print(f"Point ID: {point.id}")
    print(f"Vector values (first 5 dims): {point.vector[:5]}...")  # Vectors are lists of floats; slice to avoid spam
    print(f"Payload: {point.payload}")
    print("---")
