"""
diagnose.py
-----------
Run this to find EXACTLY where the 90 seconds is being spent.
Each step is timed independently.

Usage:
  .venv\Scripts\Activate.ps1
  python diagnose.py
"""

import time
import httpx
import asyncio

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "qwen2.5-coder:7b-instruct-q4_K_M"
INDEX_PATH      = r"D:\Gitrnd\phonex"

print("=" * 60)
print("phonex — Diagnostics")
print("=" * 60)


# ── Test 1: Can we reach Ollama at all? ──────────────────────
print("\n[1] Pinging Ollama...")
t = time.time()
try:
    r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    elapsed = time.time() - t
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"    ✅ Ollama reachable in {elapsed:.1f}s")
    print(f"    Loaded models: {models}")
    if LLM_MODEL not in models and not any(LLM_MODEL in m for m in models):
        print(f"    ⚠️  WARNING: {LLM_MODEL} not in model list!")
        print(f"    Run: ollama pull {LLM_MODEL}")
except Exception as e:
    print(f"    ❌ Cannot reach Ollama: {e}")
    print("    Make sure Ollama is running: ollama serve")
    exit(1)


# ── Test 2: How long does Ollama take to return the FIRST TOKEN? ──
print(f"\n[2] Timing first token from {LLM_MODEL}...")
print("    (This includes model load time if not already in RAM)")

async def time_first_token():
    payload = {
        "model": LLM_MODEL,
        "stream": True,
        "keep_alive": "10m",
        "options": {"num_ctx": 512, "num_thread": 12, "num_gpu": 1},
        "messages": [
            {"role": "user", "content": "Say the word hello and nothing else."}
        ],
    }
    t_start = time.time()
    first_token_time = None
    full_response = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                import json
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token and first_token_time is None:
                    first_token_time = time.time() - t_start
                    print(f"    ✅ First token in {first_token_time:.1f}s  →  '{token}'")
                full_response += token
                if chunk.get("done"):
                    break

    total = time.time() - t_start
    print(f"    Full response in {total:.1f}s  →  '{full_response.strip()}'")
    return first_token_time, total

first_tok, total_tok = asyncio.run(time_first_token())


# ── Test 3: How long does ChromaDB retrieval take? ──────────
print("\n[3] Timing ChromaDB retrieval...")
t = time.time()
try:
    import chromadb
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_BASE_URL,
    )

    chroma_client     = chromadb.PersistentClient(path=INDEX_PATH)
    chroma_collection = chroma_client.get_collection("cdpdevops")
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    index             = VectorStoreIndex.from_vector_store(vector_store)
    retriever         = index.as_retriever(similarity_top_k=2)

    t_retrieve = time.time()
    nodes = retriever.retrieve("What http classes are available?")
    elapsed = time.time() - t_retrieve
    setup_elapsed = t_retrieve - t

    print(f"    ✅ Index loaded in {setup_elapsed:.1f}s")
    print(f"    ✅ Retrieval in {elapsed:.1f}s  →  {len(nodes)} chunks found")
    for n in nodes:
        print(f"       • {n.metadata.get('file_path','?')}  score={round(n.score,3) if n.score else '?'}")
except Exception as e:
    print(f"    ❌ Retrieval failed: {e}")


# ── Test 4: End-to-end — hit the live /chat/stream endpoint ──
print("\n[4] Timing full /chat/stream endpoint (server must be running)...")

async def time_stream_endpoint():
    payload = {"message": "What http classes are available?", "reset_memory": False}
    t_start = time.time()
    first_token_time = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:8321/chat/stream",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status_code != 200:
                    print(f"    ❌ Server returned {resp.status_code}")
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    import json
                    chunk = json.loads(line[5:].strip())
                    if chunk.get("error"):
                        print(f"    ❌ Server error: {chunk['error']}")
                        return
                    if chunk.get("token"):
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - t_start
                            print(f"    ✅ First token from server in {first_token_time:.1f}s")
                    if chunk.get("done"):
                        break
        total = time.time() - t_start
        print(f"    ✅ Full response in {total:.1f}s  ({token_count} tokens)")
    except httpx.ConnectError:
        print("    ⚠️  Server not running — start python server.py first to test this step")
    except Exception as e:
        print(f"    ❌ Error: {e}")

asyncio.run(time_stream_endpoint())


# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if first_tok is not None:
    if first_tok < 5:
        print(f"  Ollama first token : {first_tok:.1f}s  ✅ FAST")
    elif first_tok < 20:
        print(f"  Ollama first token : {first_tok:.1f}s  ⚠️  SLOW (model loading from disk)")
    else:
        print(f"  Ollama first token : {first_tok:.1f}s  ❌ VERY SLOW (no GPU offload or wrong model)")
print()
print("  If Ollama is fast but /chat/stream is slow →")
print("    the retrieval step or server code is blocking the stream")
print("  If Ollama itself is slow →")
print("    run: ollama run <model> 'hello'  to pre-warm before using VS Code")
print("=" * 60)