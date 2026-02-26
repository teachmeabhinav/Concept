# phonex — Build Plan (RAG Approach)

**Project:** Local AI Code Assistant for CDP DevOps  
**Approach:** RAG (Retrieval-Augmented Generation)  
**Audience:** Junior developer (fresher) — every step is explained  
**Estimated Total Effort:** 10-14 working days  
**Created:** February 20, 2026

---

## What You Are Building

A chat application where developers type questions about the CDP DevOps .NET codebase and get accurate, codebase-aware answers. The AI runs **locally** — no data leaves the machine.

**Example interactions:**
- *"How do I add a new DataAccess client for a new external API?"*
- *"What's the difference between Routines.Core and Routines.Apis?"*
- *"Show me the request flow from API endpoint to database"*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     phonex                         │
│                                                             │
│  ┌────────────────┐    ┌───────────┐    ┌──────────────┐   │
│  │  VS Code Chat   │───▶│ RAG API    │───▶│  Local LLM   │   │
│  │  @phonex│   │ (FastAPI)  │    │  (Ollama)    │   │
│  └────────────────┘    └─────┬─────┘    └──────────────┘   │
│                              │                              │
│  ┌────────────────┐    ┌─────▼─────┐                       │
│  │  VS Code        │    │ Vector DB  │                       │
│  │  Extension      │    │ (ChromaDB) │                       │
│  │  (TypeScript)   │    └─────┬─────┘                       │
│  └────────────────┘          │                              │
│                    ┌─────────▼──────────┐                  │
│                    │  Indexed Codebase   │                  │
│                    │  (.cs, .md, .csproj)│                  │
│                    └────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

**How a question gets answered:**
1. Developer types `@phonex how does Redis caching work?` in VS Code Chat
2. VS Code extension sends the question to the local RAG API (FastAPI on localhost)
3. RAG API converts the question into a vector and searches ChromaDB for relevant code/doc chunks
4. RAG API sends the question + retrieved chunks to Ollama (local LLM)
5. LLM generates an answer grounded in the actual codebase
6. Answer streams back into the VS Code Chat panel with source file references

---

## Prerequisites

Before starting, ensure you have:

- [ ] Windows 10/11 machine with at least **16 GB RAM**
- [ ] Python 3.10 or later installed ([python.org](https://www.python.org/downloads/))
- [ ] Git installed
- [ ] VS Code installed
- [ ] Access to `C:\gitrnd\cdp-devops` repository (cloned)
- [ ] Basic Python knowledge (variables, functions, pip install)
- [ ] Node.js 18+ installed ([nodejs.org](https://nodejs.org/)) — needed for VS Code extension
- [ ] Basic TypeScript knowledge (for the VS Code extension part)

**Nice to have (not required):**
- NVIDIA GPU with 8+ GB VRAM (makes responses faster, but CPU works too)

---

## Phase 1: Set Up the Foundation (Day 1-2)

### Step 1.1: Install Ollama (Local LLM Runner)

Ollama is a tool that downloads and runs AI models on your machine.

1. Go to [ollama.com](https://ollama.com) and download the Windows installer
2. Run the installer — it installs as a system service
3. Open a terminal and verify:
   ```powershell
   ollama --version
   ```
4. Download the code-specialized model:
   ```powershell
   ollama pull deepseek-coder-v2:16b
   ```
   - This downloads ~10 GB. Wait for it to complete.
   - If your machine has less than 16 GB RAM, use the smaller model instead:
     ```powershell
     ollama pull qwen2.5-coder:7b
     ```
5. Download the embedding model (used to convert text → vectors):
   ```powershell
   ollama pull nomic-embed-text
   ```
6. Test that it works:
   ```powershell
   ollama run deepseek-coder-v2:16b "What is dependency injection in C#?"
   ```
   - You should see a response. Press Ctrl+D to exit.

**What you now have:** A local AI model running on your machine that can answer general coding questions. It doesn't know about CDP DevOps yet — that's what we build next.

---

### Step 1.2: Create the Python Project

1. Create a project folder:
   ```powershell
   mkdir C:\gitrnd\phonex
   cd C:\gitrnd\phonex
   ```

2. Create a virtual environment (isolates dependencies):
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   - You should see `(.venv)` in your terminal prompt.

3. Create a `requirements.txt` file with these contents:
   ```
   llama-index==0.12.2
   llama-index-llms-ollama==0.5.0
   llama-index-embeddings-ollama==0.5.0
   llama-index-vector-stores-chroma==0.4.1
   chromadb==0.6.3
   fastapi==0.115.0
   uvicorn==0.34.0
   pydantic==2.10.0
   ```
   > **Note:** Version numbers are current as of Feb 2026. If installation fails, try removing version pins.
   > 
   > We use **FastAPI** instead of Gradio because the chat UI is VS Code — FastAPI serves as the backend API that the VS Code extension calls.

4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   - This takes 3-5 minutes. Wait for it to complete.

**What you now have:** A Python project with all libraries needed to build the RAG pipeline.

---

## Phase 2: Index the Codebase (Day 3-4)

This is the most important phase. You're converting the CDP DevOps codebase into a searchable knowledge base.

### Step 2.1: Create the Indexing Script

Create a file `C:\gitrnd\phonex\index_codebase.py`:

```python
"""
index_codebase.py
-----------------
Reads all code and docs from cdp-devops and stores them in a vector database.
Run this ONCE (or whenever the codebase changes significantly).
"""

import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# ──────────────────────────────────────────────
# CONFIGURATION — Change these paths if needed
# ──────────────────────────────────────────────
CODEBASE_DIRS = [
    r"C:\gitrnd\cdp-devops\packages\dotnet",
    r"C:\gitrnd\cdp-devops\docs",
]
INDEX_STORAGE_DIR = r"C:\gitrnd\phonex\vector_store"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-coder-v2:16b"    # Change to "qwen2.5-coder:7b" for lower RAM

# File types to index
CODE_EXTENSIONS = [".cs", ".csproj", ".sln", ".json", ".xml"]
DOC_EXTENSIONS = [".md"]
ALL_EXTENSIONS = CODE_EXTENSIONS + DOC_EXTENSIONS


def main():
    print("=" * 60)
    print("phonex — Codebase Indexer")
    print("=" * 60)

    # ── Step 1: Configure the embedding model ──
    print("\n[1/4] Configuring embedding model...")
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120)

    # ── Step 2: Load documents from the codebase ──
    print("[2/4] Loading documents from codebase...")
    print(f"      Scanning: {CODEBASE_DIRS}")
    print(f"      File types: {ALL_EXTENSIONS}")

    documents = []
    for dir_path in CODEBASE_DIRS:
        if not os.path.exists(dir_path):
            print(f"      WARNING: Directory not found: {dir_path}")
            continue
        reader = SimpleDirectoryReader(
            input_dir=dir_path,
            recursive=True,
            required_exts=ALL_EXTENSIONS,
            filename_as_id=True,
        )
        docs = reader.load_data()
        print(f"      Loaded {len(docs)} files from {dir_path}")
        documents.extend(docs)

    if not documents:
        print("ERROR: No documents found. Check your paths.")
        return

    print(f"      Total documents loaded: {len(documents)}")

    # ── Step 3: Split documents into chunks ──
    print("[3/4] Splitting documents into searchable chunks...")

    # Use different splitters for code vs docs
    code_splitter = CodeSplitter(
        language="c_sharp",
        chunk_lines=80,         # ~80 lines per chunk
        chunk_lines_overlap=15, # 15 lines overlap between chunks
        max_chars=3000,
    )
    doc_splitter = SentenceSplitter(
        chunk_size=1024,     # ~1024 characters per chunk
        chunk_overlap=200,   # 200 char overlap
    )

    code_docs = [d for d in documents if any(d.metadata.get("file_path", "").endswith(ext) for ext in CODE_EXTENSIONS)]
    md_docs = [d for d in documents if d.metadata.get("file_path", "").endswith(".md")]

    code_nodes = code_splitter.get_nodes_from_documents(code_docs) if code_docs else []
    doc_nodes = doc_splitter.get_nodes_from_documents(md_docs) if md_docs else []
    all_nodes = code_nodes + doc_nodes

    print(f"      Code chunks: {len(code_nodes)}")
    print(f"      Doc chunks:  {len(doc_nodes)}")
    print(f"      Total chunks: {len(all_nodes)}")

    # ── Step 4: Store in vector database ──
    print("[4/4] Embedding and storing in vector database...")
    print("      (This may take 15-60 minutes depending on codebase size)")

    # Create ChromaDB persistent storage
    os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=INDEX_STORAGE_DIR)

    # Delete old collection if re-indexing
    try:
        chroma_client.delete_collection("cdpdevops")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection("cdpdevops")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index (this is where embedding happens)
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print(f"Vector store saved to: {INDEX_STORAGE_DIR}")
    print(f"Total chunks indexed: {len(all_nodes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### Step 2.2: Run the Indexer

```powershell
cd C:\gitrnd\phonex
.\.venv\Scripts\Activate.ps1
python index_codebase.py
```

- **Expected time:** 15-60 minutes (depends on codebase size and machine speed)
- **Expected output:** You'll see progress as files are loaded, chunked, and embedded
- **Result:** A `vector_store/` folder is created with the searchable database

### Step 2.3: Verify the Index

Create a quick test file `C:\gitrnd\phonex\test_query.py`:

```python
"""Quick test to verify the index is working."""

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="deepseek-coder-v2:16b", request_timeout=120)

# Load existing index
chroma_client = chromadb.PersistentClient(path=r"C:\gitrnd\phonex\vector_store")
chroma_collection = chroma_client.get_collection("cdpdevops")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Test query
engine = index.as_query_engine(similarity_top_k=5)
response = engine.query("What packages are available for data access?")
print(response)
```

Run it:
```powershell
python test_query.py
```

If you get a relevant answer mentioning DynamoDB, Redis, Postgres, etc., your index is working.

---

## Phase 3: Build the VS Code Chat Integration (Day 5-8)

This phase has two parts:
- **Part A** — A Python FastAPI server that exposes your RAG pipeline as an API
- **Part B** — A VS Code extension that adds `@phonex` to VS Code Chat

---

### Part A: RAG API Server (Python)

### Step 3.1: Create the API Server

Create `C:\gitrnd\phonex\server.py`:

```python
"""
server.py
---------
phonex RAG API Server
Exposes the RAG pipeline as a REST API that the VS Code extension calls.
Runs on http://localhost:8321
"""

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-coder-v2:16b"
INDEX_PATH = r"C:\gitrnd\phonex\vector_store"
TOP_K = 8  # Number of code chunks to retrieve per question
SERVER_PORT = 8321

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
# Initialize RAG engine
# ──────────────────────────────────────────────
print("Initializing phonex RAG server...")

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=180,
    system_prompt=SYSTEM_PROMPT,
)

chroma_client = chromadb.PersistentClient(path=INDEX_PATH)
chroma_collection = chroma_client.get_collection("cdpdevops")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Chat engine with conversation memory
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    similarity_top_k=TOP_K,
    system_prompt=SYSTEM_PROMPT,
)

print(f"RAG server ready on port {SERVER_PORT}")

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(title="phonex", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    reset_memory: bool = False  # Set to True to start a new conversation


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    """Health check — VS Code extension calls this to verify the server is running."""
    return {"status": "ok", "model": LLM_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a question and get an answer with source references."""
    if request.reset_memory:
        chat_engine.reset()

    response = chat_engine.chat(request.message)

    # Extract source file references
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file": node.metadata.get("file_path", "unknown"),
            "score": round(node.score, 3) if node.score else None,
            "preview": node.text[:200] if node.text else "",
        })

    return ChatResponse(
        answer=str(response),
        sources=sources[:5],  # Top 5 sources
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream the answer token-by-token (for real-time typing effect in VS Code)."""
    if request.reset_memory:
        chat_engine.reset()

    streaming_response = chat_engine.stream_chat(request.message)

    def generate():
        for token in streaming_response.response_gen:
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Send sources at the end
        sources = []
        for node in streaming_response.source_nodes:
            sources.append({
                "file": node.metadata.get("file_path", "unknown"),
                "score": round(node.score, 3) if node.score else None,
            })
        yield f"data: {json.dumps({'done': True, 'sources': sources[:5]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/reset")
def reset_memory():
    """Clear conversation memory to start fresh."""
    chat_engine.reset()
    return {"status": "memory cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
```

### Step 3.2: Test the API Server

```powershell
cd C:\gitrnd\phonex
.\.venv\Scripts\Activate.ps1
python server.py
```

In another terminal, test it:
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8321/health"

# Ask a question
$body = @{ message = "What packages handle data access?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8321/chat" -Method POST -Body $body -ContentType "application/json"
```

If you see a relevant answer, the API is working. Leave the server running.

---

### Part B: VS Code Extension

### Step 3.3: Scaffold the VS Code Extension

1. Install VS Code extension tooling:
   ```powershell
   npm install -g yo generator-code @vscode/vsce
   ```

2. Create the extension project:
   ```powershell
   mkdir C:\gitrnd\phonex-vscode
   cd C:\gitrnd\phonex-vscode
   ```

3. Create `package.json`:
   ```json
   {
     "name": "phonex",
     "displayName": "phonex - CDP DevOps AI Assistant",
     "description": "AI-powered code assistant for the CDP DevOps .NET codebase",
     "version": "1.0.0",
     "engines": { "vscode": "^1.93.0" },
     "categories": ["AI", "Chat"],
     "activationEvents": [],
     "main": "./out/extension.js",
     "contributes": {
       "chatParticipants": [
         {
           "id": "phonex.chat",
           "fullName": "phonex",
           "name": "phonex",
           "description": "Ask questions about the CDP DevOps .NET codebase",
           "isSticky": true,
           "commands": [
             {
               "name": "flow",
               "description": "Explain a request flow or data pipeline"
             },
             {
               "name": "pattern",
               "description": "Show a code pattern or best practice"
             },
             {
               "name": "package",
               "description": "Get info about a specific NuGet package"
             },
             {
               "name": "reset",
               "description": "Clear conversation memory and start fresh"
             }
           ]
         }
       ]
     },
     "scripts": {
       "vscode:prepublish": "npm run compile",
       "compile": "tsc -p ./",
       "watch": "tsc -watch -p ./"
     },
     "devDependencies": {
       "@types/vscode": "^1.93.0",
       "@types/node": "^20.0.0",
       "typescript": "^5.5.0"
     }
   }
   ```

4. Create `tsconfig.json`:
   ```json
   {
     "compilerOptions": {
       "module": "commonjs",
       "target": "ES2022",
       "outDir": "out",
       "lib": ["ES2022"],
       "sourceMap": true,
       "rootDir": "src",
       "strict": true
     },
     "exclude": ["node_modules", ".vscode-test"]
   }
   ```

5. Install dependencies:
   ```powershell
   npm install
   ```

### Step 3.4: Create the Extension Source Code

Create `C:\gitrnd\phonex-vscode\src\extension.ts`:

```typescript
import * as vscode from 'vscode';

// ──────────────────────────────────────────────
// CONFIGURATION
// ──────────────────────────────────────────────
const RAG_API_URL = 'http://localhost:8321';

// ──────────────────────────────────────────────
// Helper: Call the RAG API
// ──────────────────────────────────────────────
async function queryRAG(
    message: string,
    resetMemory: boolean = false
): Promise<{ answer: string; sources: Array<{ file: string; score: number | null }> }> {
    const response = await fetch(`${RAG_API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, reset_memory: resetMemory }),
    });

    if (!response.ok) {
        throw new Error(`RAG API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

async function checkHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${RAG_API_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
}

// ──────────────────────────────────────────────
// Chat Participant Handler
// ──────────────────────────────────────────────
const handler: vscode.ChatRequestHandler = async (
    request: vscode.ChatRequest,
    context: vscode.ChatContext,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken
): Promise<vscode.ChatResult> => {

    // Handle /reset command
    if (request.command === 'reset') {
        await fetch(`${RAG_API_URL}/reset`, { method: 'POST' });
        stream.markdown('Conversation memory cleared. Ask a new question!');
        return {};
    }

    // Check if the RAG server is running
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        stream.markdown(
            '**phonex server is not running.**\n\n' +
            'Start it with:\n```\ncd C:\\gitrnd\\phonex\n' +
            '.\\.venv\\Scripts\\Activate.ps1\n' +
            'python server.py\n```'
        );
        return {};
    }

    // Build the prompt based on the command
    let prompt = request.prompt;
    if (request.command === 'flow') {
        prompt = `Explain the data flow or request lifecycle for: ${request.prompt}`;
    } else if (request.command === 'pattern') {
        prompt = `Show the code pattern or best practice for: ${request.prompt}`;
    } else if (request.command === 'package') {
        prompt = `Provide details about this NuGet package: ${request.prompt}. Include its purpose, key classes, dependencies, and usage examples.`;
    }

    // Show progress
    stream.progress('Searching codebase and generating answer...');

    try {
        // Call the RAG API
        const result = await queryRAG(prompt);

        if (token.isCancellationRequested) {
            return {};
        }

        // Stream the answer
        stream.markdown(result.answer);

        // Show source references
        if (result.sources && result.sources.length > 0) {
            stream.markdown('\n\n---\n**Sources:**');
            for (const source of result.sources) {
                const score = source.score ? ` (relevance: ${source.score})` : '';
                stream.markdown(`\n- \`${source.file}\`${score}`);
            }
        }
    } catch (error: any) {
        stream.markdown(`**Error:** ${error.message}`);
    }

    return {};
};

// ──────────────────────────────────────────────
// Extension Activation
// ──────────────────────────────────────────────
export function activate(extensionContext: vscode.ExtensionContext) {
    // Register the chat participant
    const participant = vscode.chat.createChatParticipant(
        'phonex.chat',
        handler
    );

    participant.iconPath = vscode.Uri.joinPath(
        extensionContext.extensionUri,
        'icon.png'
    );

    extensionContext.subscriptions.push(participant);

    console.log('phonex chat participant activated');
}

export function deactivate() {}
```

### Step 3.5: Add an Icon

Create or download a 128x128 PNG icon and save it as:
```
C:\gitrnd\phonex-vscode\icon.png
```

You can use any image — a phoenix, the team logo, etc.

### Step 3.6: Build and Install the Extension

```powershell
cd C:\gitrnd\phonex-vscode

# Compile TypeScript
npm run compile

# Package as a .vsix file (installable extension)
vsce package

# Install in VS Code
code --install-extension phonex-1.0.0.vsix
```

Restart VS Code after installation.

### Step 3.7: Use It!

1. Make sure the RAG server is running (`python server.py` in another terminal)
2. Open VS Code → open the Chat panel (Ctrl+Shift+I or click the chat icon)
3. Type `@phonex` and ask a question:

```
@phonex What packages handle data access?

@phonex /flow How does a request go from API endpoint to database?

@phonex /pattern How do I add a new DataAccess client?

@phonex /package Routines.Core

@phonex /reset
```

The answers appear directly in VS Code Chat, with source file references.

---

## Phase 4: Improve Quality (Day 7-8)

Once the basic version works, improve the answer quality with these enhancements.

### Step 4.1: Add Metadata Enrichment

The AI gives better answers when it knows *what kind of file* each chunk came from. Create `C:\gitrnd\phonex\metadata_enricher.py`:

```python
"""
metadata_enricher.py
--------------------
Adds useful metadata to each document before indexing.
Import this in index_codebase.py to improve retrieval quality.
"""


def enrich_document_metadata(document):
    """Add structured metadata based on file path and type."""
    file_path = document.metadata.get("file_path", "")

    # Determine the package area
    if "\\Cloud\\" in file_path:
        document.metadata["package_area"] = "Cloud"
    elif "\\Rules\\" in file_path:
        document.metadata["package_area"] = "Rules"
    elif "\\ApiTesting\\" in file_path:
        document.metadata["package_area"] = "ApiTesting"
    elif "\\docs\\" in file_path:
        document.metadata["package_area"] = "Documentation"

    # Determine the layer
    if "\\src\\" in file_path:
        document.metadata["layer"] = "source"
    elif "\\tests\\" in file_path:
        document.metadata["layer"] = "test"
    elif "\\docs\\" in file_path:
        document.metadata["layer"] = "documentation"

    # Determine the component type from project name
    path_lower = file_path.lower()
    if "abstractions" in path_lower:
        document.metadata["component_type"] = "abstraction"
    elif "dataaccess" in path_lower:
        document.metadata["component_type"] = "data-access"
    elif "hosting" in path_lower:
        document.metadata["component_type"] = "hosting"
    elif "routines" in path_lower:
        document.metadata["component_type"] = "routines"
    elif "appservices" in path_lower:
        document.metadata["component_type"] = "app-services"
    elif "security" in path_lower:
        document.metadata["component_type"] = "security"
    elif "bundles" in path_lower:
        document.metadata["component_type"] = "bundle"

    return document
```

Then in `index_codebase.py`, after loading documents, add:
```python
from metadata_enricher import enrich_document_metadata

# After: documents.extend(docs)
documents = [enrich_document_metadata(doc) for doc in documents]
```

### Step 4.2: Add Follow-Up Context to the Extension

The `server.py` already includes conversation memory via `ChatMemoryBuffer`. This means developers can ask follow-up questions naturally:

- *"How does Redis caching work?"*
- *"Show me an example"* ← it remembers the context is Redis
- *"What about error handling?"* ← still in Redis context

The `/reset` command clears this memory when they want to switch topics.

### Step 4.3: Add Streaming Responses (Optional Enhancement)

For a more responsive feel (answers appear word-by-word instead of all at once), update the extension to use the `/chat/stream` endpoint.

In `extension.ts`, replace the `queryRAG` call in the handler with:

```typescript
// Stream response token-by-token
const response = await fetch(`${RAG_API_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: prompt }),
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

if (reader) {
    while (true) {
        const { done, value } = await reader.read();
        if (done || token.isCancellationRequested) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

        for (const line of lines) {
            const data = JSON.parse(line.slice(6));
            if (data.token) {
                stream.markdown(data.token);
            }
            if (data.done && data.sources) {
                stream.markdown('\n\n---\n**Sources:**');
                for (const source of data.sources) {
                    stream.markdown(`\n- \`${source.file}\``);
                }
            }
        }
    }
}
```

This is optional — the non-streaming version works fine, it just shows the full answer at once.

---

## Phase 5: Package for Distribution (Day 9-10)

Make it easy for other developers to use.

### Step 5.1: Create a Startup Script

Create `C:\gitrnd\phonex\start.cmd`:

```batch
@echo off
echo ========================================
echo   phonex RAG Server
echo ========================================
echo.
echo Checking Ollama...
ollama list >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama is not running. Please start Ollama first.
    echo Download from: https://ollama.com
    pause
    exit /b 1
)
echo Ollama OK.
echo.
echo Starting RAG API server on http://localhost:8321 ...
echo Use @phonex in VS Code Chat to ask questions.
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python server.py
pause
```

### Step 5.2: Create a Setup Script

Create `C:\gitrnd\phonex\setup.cmd`:

```batch
@echo off
echo ========================================
echo   phonex - First Time Setup
echo ========================================
echo.
echo [1/7] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo.
echo [2/7] Checking Node.js...
node --version
if errorlevel 1 (
    echo ERROR: Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
echo.
echo [3/7] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo.
echo [4/7] Installing Python dependencies...
pip install -r requirements.txt
echo.
echo [5/7] Pulling AI models (this downloads ~12 GB)...
ollama pull deepseek-coder-v2:16b
ollama pull nomic-embed-text
echo.
echo [6/7] Indexing codebase (this may take 30-60 minutes)...
python index_codebase.py
echo.
echo [7/7] Building and installing VS Code extension...
cd /d "%~dp0..\phonex-vscode"
call npm install
call npm run compile
call npx vsce package
for %%f in (*.vsix) do code --install-extension "%%f"
cd /d "%~dp0"
echo.
echo ========================================
echo   Setup complete!
echo   1. Run start.cmd to start the RAG server
echo   2. Restart VS Code
echo   3. Use @phonex in VS Code Chat
echo ========================================
pause
```

### Step 5.3: Create a README

Create `C:\gitrnd\phonex\README.md`:

```markdown
# phonex

Local AI assistant for the CDP DevOps .NET codebase, integrated into VS Code Chat.

## Quick Start

1. Install [Ollama](https://ollama.com), [Python 3.10+](https://python.org), and [Node.js 18+](https://nodejs.org)
2. Run `setup.cmd` (first time only — takes ~30-60 min)
3. Run `start.cmd` (starts the RAG API server)
4. Open VS Code → Chat panel → type `@phonex` and ask a question

## Usage in VS Code Chat

```
@phonex What packages handle data access?
@phonex /flow How does a request reach the database?
@phonex /pattern How do I add a new DataAccess client?
@phonex /package Routines.Core
@phonex /reset
```

## Re-indexing

If the codebase has changed significantly, re-index:
```
.venv\Scripts\activate
python index_codebase.py
```
Then restart the server (`start.cmd`).

## System Requirements

- Windows 10/11
- 16 GB RAM (minimum), 32 GB recommended
- ~15 GB disk space (models + index)
- Optional: NVIDIA GPU with 8+ GB VRAM for faster responses
```

---

## Phase 6: Testing & Validation (Day 11-12)

### Step 6.1: Create a Test Suite

Create `C:\gitrnd\phonex\test_quality.py`:

```python
"""
test_quality.py
---------------
Runs a set of known questions and checks that answers contain expected keywords.
This is NOT a unit test — it's a quality validation tool.
"""

TEST_CASES = [
    {
        "question": "What packages handle database access?",
        "expected_keywords": ["DataAccess", "Postgres", "DynamoDb", "Redis"],
        "min_keywords": 2,  # At least 2 of the 4 should appear
    },
    {
        "question": "How do I add a new API endpoint?",
        "expected_keywords": ["Routines", "Hosting", "Controller", "Apis"],
        "min_keywords": 2,
    },
    {
        "question": "What security options are available?",
        "expected_keywords": ["KeyCloak", "Okta", "Security", "Authentication"],
        "min_keywords": 2,
    },
    {
        "question": "How does configuration work?",
        "expected_keywords": ["AppServices", "Configuration", "Aws", "SSM"],
        "min_keywords": 2,
    },
    {
        "question": "What is the Routines pipeline?",
        "expected_keywords": ["Routines.Core", "Hosting", "Abstractions"],
        "min_keywords": 2,
    },
]


def run_tests(query_engine):
    """Run all test cases and report results."""
    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}/{len(TEST_CASES)}: {test['question']}")
        response = str(query_engine.query(test["question"]))

        found = [kw for kw in test["expected_keywords"] if kw.lower() in response.lower()]
        missing = [kw for kw in test["expected_keywords"] if kw.lower() not in response.lower()]

        if len(found) >= test["min_keywords"]:
            print(f"  PASS — Found: {found}")
            passed += 1
        else:
            print(f"  FAIL — Found: {found}, Missing: {missing}")
            print(f"  Response preview: {response[:200]}...")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    # Same initialization as app.py
    import chromadb
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(model="deepseek-coder-v2:16b", request_timeout=180)

    chroma_client = chromadb.PersistentClient(path=r"C:\gitrnd\phonex\vector_store")
    chroma_collection = chroma_client.get_collection("cdpdevops")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(similarity_top_k=8)
    run_tests(query_engine)
```

### Step 6.2: Validation Checklist

Run through this checklist before sharing with the team:

- [ ] `setup.cmd` completes without errors on a clean machine
- [ ] `start.cmd` launches the RAG API server (http://localhost:8321/health returns OK)
- [ ] VS Code extension installs without errors
- [ ] `@phonex` appears in VS Code Chat participant list
- [ ] All 5 example questions return relevant answers
- [ ] Answers reference actual class/project names from cdp-devops
- [ ] Source citations show correct file paths
- [ ] Response time is under 30 seconds per question
- [ ] `/reset` command clears conversation memory
- [ ] Extension shows friendly error message when server is not running
- [ ] `test_quality.py` passes at least 4 out of 5 tests

---

## Final File Structure

When complete, your project should look like this:

```
C:\gitrnd\phonex\              # RAG Backend (Python)
├── .venv\                              # Python virtual environment (auto-created)
├── vector_store\                       # ChromaDB index (auto-created by indexer)
├── server.py                           # RAG API server - FastAPI (Phase 3A)
├── index_codebase.py                   # Codebase indexer (Phase 2)
├── metadata_enricher.py                # Metadata enrichment (Phase 4)
├── test_query.py                       # Quick index test (Phase 2)
├── test_quality.py                     # Quality validation (Phase 6)
├── requirements.txt                    # Python dependencies (Phase 1)
├── setup.cmd                           # First-time setup script (Phase 5)
├── start.cmd                           # Startup script — starts RAG server (Phase 5)
└── README.md                           # User documentation (Phase 5)

C:\gitrnd\phonex-vscode\       # VS Code Extension (TypeScript)
├── src\
│   └── extension.ts                    # Chat participant handler (Phase 3B)
├── out\                                # Compiled JS (auto-created by tsc)
├── package.json                        # Extension manifest + chat commands
├── tsconfig.json                       # TypeScript config
├── icon.png                            # Extension icon (128x128)
└── phonex-1.0.0.vsix          # Packaged extension (auto-created by vsce)
```

---

## Troubleshooting Guide

| Problem | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed | Download from ollama.com |
| `Error: model not found` | Model not downloaded | Run `ollama pull deepseek-coder-v2:16b` |
| Out of memory error | Model too large for RAM | Switch to `qwen2.5-coder:7b` |
| Indexing takes hours | Large codebase | Reduce `ALL_EXTENSIONS` to just `.cs` and `.md` |
| Answers are generic/wrong | Bad chunking or low TOP_K | Increase `TOP_K` to 10-12, re-index with better splitter settings |
| `ConnectionError` to Ollama | Ollama service not running | Start Ollama from system tray or run `ollama serve` |
| Slow responses (>60s) | No GPU, using CPU | Expected on CPU — consider renting a cloud GPU for team use |
| `@phonex` not in chat list | Extension not installed | Run `code --install-extension phonex-1.0.0.vsix` and restart VS Code |
| "server is not running" in chat | RAG server not started | Run `start.cmd` or `python server.py` in a terminal |
| `npm run compile` fails | TypeScript errors | Check Node.js version (need 18+), run `npm install` again |
| `vsce package` fails | Missing fields | Ensure `package.json` has publisher field: add `"publisher": "cdp-team"` |

---

## Future Enhancements (After MVP)

Once the basic version works, consider these improvements:

1. **Streaming Responses** — Token-by-token streaming for real-time typing effect (see Step 4.3)
2. **Auto Re-indexing** — Watch for file changes and update the index automatically
3. **Team Server** — Deploy on a shared machine so everyone queries the same instance
4. **Fine-tuning** — Create 500+ Q&A pairs and fine-tune the base model for even better answers
5. **Multi-repo** — Index additional repos (cdp-services, cdp-infrastructure, etc.)
6. **Feedback Loop** — Add thumbs up/down buttons and use feedback to improve the system prompt

---

*Plan created: February 20, 2026*
*Project codename: phonex*
