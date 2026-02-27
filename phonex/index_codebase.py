"""
index_codebase.py
-----------------
Reads all code and docs from cdp-devops and stores them in a vector database.
Run this ONCE (or whenever the codebase changes significantly).
"""

import os
import shutil
import chromadb
import tempfile
import zipfile
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Detect if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# ──────────────────────────────────────────────
# CONFIGURATION — Change these paths if needed
# ──────────────────────────────────────────────
CODEBASE_DIRS = [
    r"D:\Gitrnd\dotnet\src\aspnetcore\src\Http"
]
# Local data folder — drop any files/folders here and they will be indexed too.
# In Colab this maps to /content/phonex/data (auto-detected below).
_local_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.isdir(_local_data) and _local_data not in CODEBASE_DIRS:
    CODEBASE_DIRS.append(_local_data)

# Default index storage (will use local /tmp/phonex_index in Colab or ./phonex_index locally)
if IN_COLAB:
    INDEX_STORAGE_DIR = "/tmp/phonex_index"
else:
    INDEX_STORAGE_DIR = "./phonex_index"
# Embedding model name for SentenceTransformer
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# All files under the codebase directories are indexed — no extension filter.
# Binary files that llama_index cannot parse are automatically skipped.
SKIP_EXTENSIONS = {'.exe', '.dll', '.pdb', '.obj', '.bin', '.png', '.jpg',
                   '.jpeg', '.gif', '.ico', '.svg', '.ttf', '.woff', '.woff2',
                   '.eot', '.mp4', '.zip', '.tar', '.gz', '.7z', '.pdf'}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────


def extract_if_zip(path: str) -> str:
    """If the given path is a .zip file, extract it to a temporary folder and return
    the extraction directory. Otherwise return the original path.

    The reader cannot walk inside zip archives, so we need to unpack them first. In
    Colab you can upload a zip (e.g. `aspnetcore-main.zip`) and then point
    CODEBASE_DIRS at it; this helper will unzip it automatically. The archive itself is
    not indexed, but its contents are. A subsequent run reuses the temp directory if it
    already exists.
    """
    if path.lower().endswith(".zip"):
        if not os.path.exists(path):
            return path
        base = os.path.splitext(os.path.basename(path))[0]
        temp_dir = os.path.join(tempfile.gettempdir(), base)
        if not os.path.exists(temp_dir):
            print(f"         Extracting {os.path.basename(path)}...")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(temp_dir)
            print(f"         ✓ Extracted to {temp_dir}")
        return temp_dir
    return path


def main():
    global CODEBASE_DIRS
    
    print("=" * 60)
    print("phonex — Codebase Indexer")
    print("=" * 60)

    # ── Step 1: Configure the embedding model ──
    print("\n[1/4] Configuring embedding model...")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
        Settings.embed_model = embed_model
        print(f"      ✓ Embedder ready (model: {EMBEDDING_MODEL})")
    except Exception as e:
        print(f"      ❌ ERROR: could not initialize embedder: {e}")
        return

    # ── Step 2: Load documents from the codebase ──
    print("[2/4] Loading documents from codebase...")
    
    # If in Colab, MUST auto-detect and override defaults
    if IN_COLAB:
        print("      🔍 Colab mode: auto-detecting codebase...")
        detected = []

        # The directory that contains this script — we never want to index the tool itself
        script_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))

        # Scan /content/ for zip files first (highest priority), then directories
        try:
            items = sorted(os.listdir('/content/'))
            # Zips first — user uploaded target codebase as a zip (e.g. aspnetcore-main.zip)
            for item in items:
                path = f'/content/{item}'
                if item.lower().endswith('.zip') and os.path.isfile(path):
                    detected.append(path)
                    print(f"        📦 zip: {item}")
            # Also scan inside /content/phonex/ for zips bundled inside phonex.zip
            phonex_sub = '/content/phonex'
            if os.path.isdir(phonex_sub):
                for sub_item in sorted(os.listdir(phonex_sub)):
                    sub_path = os.path.join(phonex_sub, sub_item)
                    if sub_item.lower().endswith('.zip') and os.path.isfile(sub_path):
                        detected.append(sub_path)
                        print(f"        📦 zip (in phonex/): {sub_item}")
                # Also include phonex/data/ if it exists
                data_dir = os.path.join(phonex_sub, 'data')
                if os.path.isdir(data_dir):
                    detected.append(data_dir)
                    print(f"        📁 data dir: phonex/data/")
            # Then directories, excluding system dirs and the phonex tool itself
            system_dirs = {'drive', 'sample_data', '__pycache__', '__MACOSX'}
            for item in items:
                path = f'/content/{item}'
                if (os.path.isdir(path)
                        and not item.startswith('.')
                        and item not in system_dirs
                        and os.path.realpath(path) != script_dir):
                    detected.append(path)
                    print(f"        📁 dir:  {item}")
        except Exception as e:
            print(f"      ⚠️ Error scanning /content/: {e}")

        if detected:
            CODEBASE_DIRS = detected
            print(f"      ✓ Found and using {len(detected)} source(s):")
            for d in detected:
                print(f"        - {os.path.basename(d)}")
            print(f"      CODEBASE_DIRS is now: {CODEBASE_DIRS}")
        else:
            print(f"      ⚠️ No sources found in /content/, will try defaults")
            print(f"      Items in /content/: {os.listdir('/content/')}")

    print(f"\n      📁 Will scan: {CODEBASE_DIRS}")
    print(f"      Mode: ALL files (binary/media skipped automatically)")

    documents = []
    for dir_path in CODEBASE_DIRS:
        print(f"\n      Processing: {dir_path}")
        
        # support zip files by extracting them to a temp dir first
        orig = dir_path
        dir_path = extract_if_zip(dir_path)
        if dir_path != orig:
            print(f"      → Unzipped archive to: {dir_path}")

        if not os.path.exists(dir_path):
            print(f"      ❌ ERROR: Directory not found: {dir_path}")
            continue
        
        # List what we found before scanning
        contents = os.listdir(dir_path)
        print(f"      Found {len(contents)} items in directory")

        # Count files, skipping binaries
        total_files = sum(
            1 for root, _, files in os.walk(dir_path)
            for f in files
            if os.path.splitext(f)[1].lower() not in SKIP_EXTENSIONS
        )
        print(f"      Indexable files found in tree: {total_files}")

        reader = SimpleDirectoryReader(
            input_dir=dir_path,
            recursive=True,
            exclude_hidden=True,
            filename_as_id=True,
        )
        docs = reader.load_data()
        print(f"      ✓ Loaded {len(docs)} files from {dir_path}")
        documents.extend(docs)

    if not documents:
        print("\n❌ ERROR: No documents found!")
        print("   Reasons this might happen:")
        print("   1. Codebase directory is empty or path does not exist")
        print("   2. All files were binary/unreadable")
        print("   3. ZIP extraction failed")
        return

    print(f"      Total documents loaded: {len(documents)}")

    # ── Step 3: Split documents into chunks ──
    print("[3/4] Splitting documents into searchable chunks...")

    # Single splitter for all file types
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    all_nodes = splitter.get_nodes_from_documents(documents)
    print(f"      Total chunks: {len(all_nodes)}")

    # ── Step 4: Store in vector database ──
    print("[4/4] Embedding and storing in vector database...")
    print("      (This may take 15-60 minutes depending on codebase size)")
    
    if len(all_nodes) == 0:
        print("      ⚠️ WARNING: No chunks to index!")
        return

    # Create ChromaDB persistent storage
    os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)
    print(f"      Index storage: {INDEX_STORAGE_DIR}")
    
    chroma_client = chromadb.PersistentClient(path=INDEX_STORAGE_DIR)

    # Delete old collection if re-indexing
    try:
        chroma_client.delete_collection("asp_netcore")
        print("      Cleared old collection")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection("asp_netcore")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index (this is where embedding happens)
    print(f"      Starting embedding of {len(all_nodes)} chunks...")
    try:
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        print("      ✓ Embedding complete")
    except Exception as e:
        print(f"      ❌ ERROR during embedding: {e}")
        raise

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print(f"Vector store saved to: {INDEX_STORAGE_DIR}")
    print(f"Total chunks indexed: {len(all_nodes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()