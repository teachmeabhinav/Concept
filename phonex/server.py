"""
server.py
---------
phonex RAG API Server
Runs on http://localhost:8321

Diagnosis results (Feb 2026):
  Ollama first token : 5.2s  ✅ fast
  ChromaDB retrieval : 2.2s  ✅ fast
  /chat/stream first token: 37.2s ❌ server blocking before first SSE event

Root cause (original):
  The async generator was doing retrieval (~2s) + embedding (~5s) +
  Ollama connection setup (~20s) before yielding ANYTHING to the client.

Root cause (Feb 2026 v2 — VS Code extension silent):
  SSE comment heartbeats (": heartbeat\n\n") are only ~13 bytes each.
  Node.js's fetch (Undici) has a ~16KB internal response buffer — it will
  NOT call reader.read() until that buffer fills up OR the connection closes.
  So the extension's reader.read() never resolves, firstTokenReceived stays
  false, and the 90s AbortController fires → user sees no response.

Fix:
  1. Immediately yield a real JSON data event ("status":"working") as the
     very first thing in generate(). This is a proper SSE data line that
     Undici flushes to the reader immediately, clearing firstTokenReceived
     and disarming the timeout.
  2. Keep ": heartbeat" comments during subsequent waits — they still help
     with proxies/load balancers that close idle connections.
  3. Extension ignores {"status":"working"} chunks (handled client-side).

Timeline:
  t=0s   : HTTP connection → immediate {"status":"working"} event flushed
  t=0s   : Extension reader.read() resolves, timeout cleared ✅
  t=2s   : retrieval complete (ChromaDB + embedding)
  t=5s   : Ollama starts generating, first real token sent
  t=5-70s: tokens stream to VS Code in real time
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import time
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "qwen2.5-coder:7b-instruct-q4_K_M"
INDEX_PATH      = r"D:\Gitrnd\phonex\indexfolder\phonex_index_download"
COLLECTION_NAME = "asp_netcore"
TOP_K           = 2
SERVER_PORT     = 8321

SYSTEM_PROMPT = """You are phonex, an expert AI assistant for the CDP DevOps .NET cloud platform.

You have deep knowledge of:
- Cloud packages: AppServices, Routines, DataAccess, Hosting, Security
- Rules engine: Domain, Engine, Specifications, EF Core data access
- API Testing: SpecFlow/BDD infrastructure
- Architecture: Layered design (Hosting → Routines → DataAccess → External Services)

When answering:
1. Reference specific classes, interfaces, and namespaces from the codebase
2. Show code examples that follow existing patterns in the repo
3. Explain the WHY behind architectural decisions
4. If you're not sure about something, say so — don't make up class names
5. When suggesting new code, follow the existing naming conventions and project structure

Key architectural patterns:
- Every DataAccess client has an .Abstractions project (interface) + implementation project
- Services are composed via Bundles (Common.Bundles, Routines.Bundles)
- Configuration comes from AWS SSM via AppServices.Configuration.Aws
- Security uses KeyCloak or Okta via AppServices.Security.*
- Logging goes to CloudWatch via AppServices.Logging.Aws.CloudWatch
"""


# ──────────────────────────────────────────────
# Initialize LlamaIndex (retrieval only)
# ──────────────────────────────────────────────
print("Initializing phonex RAG server...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=120,
    keep_alive="10m",
    additional_kwargs={"num_ctx": 2048, "num_thread": 12, "num_gpu": 1},
    system_prompt=SYSTEM_PROMPT,
)

chroma_client     = chromadb.PersistentClient(
    path=INDEX_PATH,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
index             = VectorStoreIndex.from_vector_store(vector_store)

memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    similarity_top_k=TOP_K,
    system_prompt=SYSTEM_PROMPT,
)

simple_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
)
chat_engine._get_response_synthesizer = lambda *args, **kwargs: simple_synthesizer

print("   Synthesizer : SIMPLE_SUMMARIZE ✅")
print(f"✅ RAG server ready → http://localhost:{SERVER_PORT}")
print(f"   Model:      {LLM_MODEL} | TOP_K: {TOP_K}")
print(f"   Embeddings: {EMBEDDING_MODEL}")
print(f"   Index:      {INDEX_PATH}  (collection: {COLLECTION_NAME})")


# ──────────────────────────────────────────────
# Pre-warm Ollama on startup (eliminates cold-start delay for first query)
# ──────────────────────────────────────────────
def _prewarm_ollama() -> None:
    """Send a tiny synchronous request to load the model into memory."""
    import httpx as _httpx
    try:
        print(f"Pre-warming {LLM_MODEL}...")
        r = _httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_ctx": 64, "num_thread": 12, "num_gpu": 1},
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=60.0,
        )
        if r.status_code == 200:
            print(f"   {LLM_MODEL} is hot and ready.")
        else:
            print(f"   Pre-warm got HTTP {r.status_code} — model will load on first query.")
    except Exception as e:
        print(f"   Pre-warm skipped ({e}) — Ollama may not be running yet.")




# ──────────────────────────────────────────────
# Retrieval helper (blocking — called via asyncio.to_thread)
# ──────────────────────────────────────────────
def retrieve_context(query: str) -> tuple[str, list[dict]]:
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(query)
    context_parts = []
    sources = []
    for node in nodes:
        context_parts.append(node.text)
        sources.append({
            "file":    node.metadata.get("file_path", "unknown"),
            "score":   round(node.score, 3) if node.score else None,
            "preview": node.text[:200],
        })
    return "\n\n---\n\n".join(context_parts), sources


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="phonex", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    reset_memory: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    total_time_s: float = 0.0


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     LLM_MODEL,
        "top_k":     TOP_K,
        "stream":    "httpx direct + SSE keepalive + immediate flush",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Blocking endpoint for PowerShell testing."""
    if request.reset_memory:
        chat_engine.reset()
    t_start = time.perf_counter()
    response = await asyncio.to_thread(chat_engine.chat, request.message)
    total_time = time.perf_counter() - t_start
    print(f"[/chat] Total response time: {total_time:.2f}s")
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file":    node.metadata.get("file_path", "unknown"),
            "score":   round(node.score, 3) if node.score else None,
            "preview": node.text[:200] if node.text else "",
        })
    return ChatResponse(answer=str(response), sources=sources[:TOP_K], total_time_s=round(total_time, 2))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    True streaming endpoint with immediate flush + SSE keepalive heartbeats.

    FIX FOR VS CODE EXTENSION SILENCE:
    Node.js fetch (Undici) buffers ~16KB before delivering data to reader.read().
    SSE comment heartbeats (": heartbeat") are ~13 bytes each — they fill the
    buffer too slowly, so reader.read() never resolves and the 90s timeout fires.

    Solution: yield a real JSON data event {"status":"working"} as the VERY FIRST
    thing. Undici flushes proper SSE data lines immediately, so the extension's
    reader.read() resolves at t=0, clearing the firstTokenReceived flag and
    disarming the AbortController before any real work begins.
    """
    if request.reset_memory:
        chat_engine.reset()

    # Queue bridges background task → async generator
    # Sentinel value None signals the generator to stop
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    t_start = time.perf_counter()

    async def generate():
        try:
            # ── CRITICAL FIX: send real JSON data event immediately ──────
            # This bypasses Node.js/Undici's 16KB buffer threshold and makes
            # reader.read() resolve instantly in the VS Code extension,
            # clearing the firstTokenReceived flag and disarming the timeout.
            yield f"data: {json.dumps({'status': 'working'})}\n\n"

            # ── Step 1: Start background task for retrieval + LLM ──────
            background = asyncio.create_task(
                run_rag_in_background(request.message, queue, loop)
            )

            # ── Step 2: Drain queue, sending heartbeats while waiting ──
            while True:
                try:
                    # Wait up to 1 second for a token
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)

                    if item is None:
                        # Sentinel — background task is done
                        break

                    # Attach total_time_s to the final done event
                    if isinstance(item, dict) and item.get("done"):
                        total_time = time.perf_counter() - t_start
                        item["total_time_s"] = round(total_time, 2)
                        print(f"[/chat/stream] Total response time: {total_time:.2f}s")

                    # Item is either a token dict or a sources/error dict
                    yield f"data: {json.dumps(item)}\n\n"

                except asyncio.TimeoutError:
                    # No token arrived in 1s — send heartbeat comment.
                    # Helps with proxies/load balancers that close idle connections.
                    # (These alone are NOT enough for Node.js Undici — hence the
                    # immediate {"status":"working"} event above.)
                    yield ": heartbeat\n\n"

            # Ensure background task is cleaned up
            await background

        except Exception as e:
            total_time = time.perf_counter() - t_start
            print(f"[/chat/stream] Error after {total_time:.2f}s: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def run_rag_in_background(
    message: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Runs in an asyncio Task. Does retrieval then streams from Ollama,
    putting each token into the queue for the SSE generator to pick up.
    """
    sources = []
    try:
        # ── Retrieval (blocking → run in thread) ──
        context, sources = await asyncio.to_thread(retrieve_context, message)

        # ── Build prompt ──
        user_prompt = (
            f"Here is relevant code from the codebase:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {message}"
        )

        # ── Stream from Ollama directly via httpx async ──
        ollama_payload = {
            "model":      LLM_MODEL,
            "stream":     True,
            "keep_alive": "10m",
            "options":    {"num_ctx": 2048, "num_thread": 12, "num_gpu": 1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_payload,
            ) as resp:
                if resp.status_code != 200:
                    await queue.put({"error": f"Ollama returned {resp.status_code}"})
                    return

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        await queue.put({"token": token})

                    if chunk.get("done", False):
                        break

        # ── Send sources then sentinel ──
        await queue.put({"done": True, "sources": sources[:TOP_K]})

    except Exception as e:
        await queue.put({"error": str(e)})

    finally:
        # Always put sentinel so generator knows to stop
        await queue.put(None)


@app.post("/reset")
def reset_memory():
    chat_engine.reset()
    return {"status": "memory cleared"}


if __name__ == "__main__":
    _prewarm_ollama()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
