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
    r"D:\Gitrnd\dotnet\src\aspnetcore\src\Http"
]
INDEX_STORAGE_DIR = r"D:\Gitrnd\phonex"
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
    code_splitter = SentenceSplitter(
        chunk_size=1024,     # ~1024 characters per chunk
        chunk_overlap=200,   # 200 char overlap
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