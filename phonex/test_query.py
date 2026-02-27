"""
test_query.py
-------------
Quick test to verify the index is working.
Optimized for i7-1270P / 32GB RAM / Intel Xe GPU (no dedicated NVIDIA card).
"""

import time
import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# OLD: model="qwen2.5-coder:7b"  (generic pull — gets q4_0 quantization)
# FIX: q4_K_M is a smarter quantization format that uses AVX2 instructions
# on your i7-1270P for faster matrix math. Same quality, ~15-20% faster than q4_0.
# Run: ollama pull qwen2.5-coder:7b-instruct-q4_K_M
LLM_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "asp_netcore"
INDEX_PATH = r"D:\Gitrnd\phonex\indexfolder\phonex_index_download"


# ──────────────────────────────────────────────
# Configure models
# ──────────────────────────────────────────────
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=60,
    keep_alive="10m",
    additional_kwargs={"num_ctx": 2048, "num_thread": 12, "num_gpu": 1},
)


# ──────────────────────────────────────────────
# Load the existing index
# ──────────────────────────────────────────────
print(f"Loading index from {INDEX_PATH}...")
chroma_client = chromadb.PersistentClient(
    path=INDEX_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)


# ──────────────────────────────────────────────
# Build query engine with speed optimizations
# ──────────────────────────────────────────────

# Use retriever directly so we can stream tokens from the LLM live.
retriever = index.as_retriever(similarity_top_k=2)


# ──────────────────────────────────────────────
# Run test query with timing
# ──────────────────────────────────────────────
QUERY = "What is the language of the code?"

print(f"\nQuery: {QUERY}")
print("-" * 50)

# Step 1: Retrieve relevant chunks
print("[Retrieving context...]")
retrieve_start = time.time()
nodes = retriever.retrieve(QUERY)
context = "\n\n---\n\n".join(n.text for n in nodes)
print(f"[Retrieved {len(nodes)} chunks in {time.time()-retrieve_start:.1f}s]\n")

# Step 2: Build prompt and stream tokens live
prompt = f"Here is relevant code from the codebase:\n\n{context}\n\n---\n\nQuestion: {QUERY}"

start = time.time()
full_text = []
for chunk in Settings.llm.stream_complete(prompt):
    token = chunk.delta
    print(token, end="", flush=True)
    full_text.append(token)

elapsed = time.time() - start
print("\n" + "-" * 50)
print(f"Response time: {elapsed:.1f} seconds")

# Give feedback on whether the speed is on target
if elapsed < 15:
    print("[OK] Speed: EXCELLENT (under 15 sec)")
elif elapsed < 30:
    print("[OK] Speed: GOOD (under 30 sec)")
elif elapsed < 60:
    print("[WARN] Speed: ACCEPTABLE -- check OLLAMA_NUM_THREAD and OLLAMA_GPU_LAYERS env vars")
else:
    print("[SLOW] Speed: TOO SLOW -- model may not be loaded, or num_thread/num_gpu not set")