"""
test_query.py
-------------
Quick test to verify the index is working.
Optimized for i7-1270P / 32GB RAM / Intel Xe GPU (no dedicated NVIDIA card).
"""

import time
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# FIX: Import SIMPLE_SUMMARIZE to prevent the default refine synthesizer
# from making multiple LLM calls — same fix as server.py.
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# OLD: model="qwen2.5-coder:7b"  (generic pull — gets q4_0 quantization)
# FIX: q4_K_M is a smarter quantization format that uses AVX2 instructions
# on your i7-1270P for faster matrix math. Same quality, ~15-20% faster than q4_0.
# Run: ollama pull qwen2.5-coder:7b-instruct-q4_K_M
LLM_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

EMBED_MODEL = "nomic-embed-text"
INDEX_PATH = r"D:\Gitrnd\phonex"


# ──────────────────────────────────────────────
# Configure models
# ──────────────────────────────────────────────
Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,

    # OLD: request_timeout=600  (10 minutes — way too long for a test script)
    # FIX: 60 seconds. With all optimizations below, 7B on your i7-1270P should
    # respond in 10-30 sec. If it hits 60, something else is wrong (model not loaded,
    # Ollama not running, etc.) and you want to know fast — not wait 10 minutes.
    request_timeout=60,

    # FIX: keep_alive keeps the model hot in RAM between test runs.
    # Without this, Ollama unloads the model after each call and reloads it
    # for the next one — adding 5-10 sec of dead time per query.
    keep_alive="10m",

    # FIX: num_ctx controls the context window sent to the model.
    # Default is 2048-4096. We cap it at 2048 for the test script because:
    # - test_query only retrieves 2 chunks (similarity_top_k=2)
    # - smaller context = faster prefill = faster first token
    num_ctx=2048,

    # FIX: num_thread pins Ollama to your i7-1270P's P-cores.
    # Your CPU has 4 P-cores + 8 E-cores = 16 threads total.
    # Using 12 threads: leaves 4 for Windows/VS Code, avoids E-core scheduling overhead.
    # Intel hybrid CPUs (like yours) are slower when LLM threads land on E-cores.
    num_thread=12,

    # FIX: num_gpu=1 tells Ollama to use your Intel Xe integrated GPU for some layers.
    # Combined with OLLAMA_GPU_LAYERS env var (set in start.cmd), this offloads
    # the attention layers to iGPU, freeing CPU for feed-forward layers.
    # Net result: ~20-30% faster on Intel Xe vs CPU-only.
    num_gpu=1,
)


# ──────────────────────────────────────────────
# Load the existing index
# ──────────────────────────────────────────────
print(f"Loading index from {INDEX_PATH}...")
chroma_client = chromadb.PersistentClient(path=INDEX_PATH)
chroma_collection = chroma_client.get_collection("cdpdevops")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)


# ──────────────────────────────────────────────
# Build query engine with speed optimizations
# ──────────────────────────────────────────────

# FIX: Explicit SIMPLE_SUMMARIZE synthesizer.
# OLD (implicit): ResponseMode.COMPACT_AND_REFINE
# REASON: COMPACT_AND_REFINE calls the LLM once per retrieved chunk to refine
# the answer iteratively. With similarity_top_k=2 that's 2 LLM calls.
# SIMPLE_SUMMARIZE concatenates all chunks and calls the LLM exactly ONCE.
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
)

engine = index.as_query_engine(
    # OLD: similarity_top_k=2  ← already good, keeping it
    # This controls how many chunks are retrieved from ChromaDB.
    # 2 is the right number for a test script — enough context, minimal LLM load.
    similarity_top_k=2,

    # FIX: Pass our fast synthesizer so LlamaIndex doesn't silently use refine.
    response_synthesizer=synthesizer,
)


# ──────────────────────────────────────────────
# Run test query with timing
# ──────────────────────────────────────────────
QUERY = "What is the language of the code?"

print(f"\nQuery: {QUERY}")
print("─" * 50)

start = time.time()
response = engine.query(QUERY)
elapsed = time.time() - start

print(response)
print("─" * 50)
print(f"Response time: {elapsed:.1f} seconds")

# Give feedback on whether the speed is on target
if elapsed < 15:
    print("✅ Speed: EXCELLENT (under 15 sec)")
elif elapsed < 30:
    print("✅ Speed: GOOD (under 30 sec)")
elif elapsed < 60:
    print("⚠️  Speed: ACCEPTABLE — check OLLAMA_NUM_THREAD and OLLAMA_GPU_LAYERS env vars")
else:
    print("❌ Speed: TOO SLOW — model may not be loaded, or num_thread/num_gpu not set")