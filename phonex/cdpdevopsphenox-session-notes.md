# CDP DevOps Phenox — Session Notes

**Date:** February 20, 2026  
**Participants:** Developer + GitHub Copilot  
**Topic:** Building an AI-powered code assistant for the `cdp-devops` codebase

---

## Session Summary

Three-part discussion exploring how to give developers AI-assisted guidance over the CDP DevOps .NET codebase (`C:\gitrnd\cdp-devops\packages\dotnet` and `C:\gitrnd\cdp-devops\docs`).

---

## Part 1: MPS Module Approach

### Goal

Create an MPS (Modular Prompt System) namespace for `cdp-devops` so that Copilot Chat becomes codebase-aware — helping developers add features, find solutions, or understand data flows.

### Target Codebase Overview

| Area | Location | Contents |
|------|----------|----------|
| Cloud packages | `packages/dotnet/Cloud/` | ~90 .NET projects — AppServices, Routines, DataAccess, Hosting, Security |
| Rules packages | `packages/dotnet/Rules/` | Rules engine — Domain, Engine, Specifications, EF Core data access |
| ApiTesting packages | `packages/dotnet/ApiTesting/` | SpecFlow/BDD infrastructure and abstractions |
| Documentation | `docs/` | Consumer guide, contributor guide, platform guide, API design, C# standards, context services, data access clients, reference docs |

### Proposed MPS Module Structure

| Module | Prefix | Purpose |
|--------|--------|---------|
| **devops** | `devops` | Root — architecture map, package taxonomy, layer relationships |
| **devops-defaults** | `devops-defaults` | Code generation defaults, naming conventions, project structure rules |
| **cloud-packages** | `cloud-pkg` | Deep knowledge of Cloud packages: Routines pipeline, AppServices, DataAccess, Hosting, Security |
| **rules-packages** | `rules-pkg` | Rules engine: domain model, engine abstractions, specification pattern, EF Core |
| **api-testing** | `api-test` | SpecFlow/BDD testing patterns, step library, test setup |
| **flow-guide** | `flow` | Request lifecycle: Hosting → Routines → DataAccess → External Services |

### How It Helps Developers

- **Adding functionality** → Extension point maps, pattern catalogs, dependency rules
- **Finding solutions** → Decision trees for choosing the right package/pattern
- **Understanding flows** → Layer dependency graph, cross-cutting concerns map

### Recommendation

Start with 2 modules (`devops` + `devops-defaults`), validate with the team, then expand.

---

## Part 2: Local AI Model Approach

### Pivot

Instead of just MPS modules (which enhance an existing AI like Copilot), explore running a **local AI model** — a self-contained assistant anyone can add to VS Code or any chat interface.

### Options Evaluated

| # | Approach | Tool | Effort | Best For |
|---|----------|------|--------|----------|
| 1 | Local RAG Pipeline | Ollama + ChromaDB + LangChain | 2-3 days | Most control |
| 2 | AnythingLLM | Desktop app (drag-and-drop) | 30 min | Quickest demo |
| 3 | Continue.dev + Ollama | VS Code extension | 1-2 hours | Best VS Code experience |
| 4 | Custom LlamaIndex Script | Python script + web UI | 1-2 days | Team sharing |
| 5 | MPS + Local RAG Hybrid | MPS system prompt + RAG search | 3-5 days | Production quality |

### RAG Pipeline Architecture

```
Developer asks question
        ↓
Embedding model converts question → vector
        ↓
Vector DB finds top-K relevant code/doc chunks
        ↓
Local LLM receives: system prompt + retrieved chunks + question
        ↓
Answer grounded in YOUR codebase
```

### What Gets Indexed

- All `*.cs` files from `packages/dotnet/` (split by class/method)
- All `*.csproj` files (dependency maps)
- All `*.md` files from `docs/`
- Solution structure metadata

---

## Part 3: Key Concepts Clarified

### Why Use a Base Model (Not Train From Scratch)?

Training an LLM from scratch requires ~$1M–$100M in compute, trillions of tokens, and months of work. The CDP DevOps codebase (~500MB) is far too small to train a model that can even form sentences.

**The correct approach is layering:**

```
┌─────────────────────────────────────────┐
│  "cdpdevopsphenox"  (your product name) │
├─────────────────────────────────────────┤
│  YOUR knowledge layer                   │  ← What you build
│  • RAG index of code + docs             │
│  • Custom system prompt                 │
│  • Fine-tuning (optional)               │
├─────────────────────────────────────────┤
│  Base LLM (DeepSeek / LLaMA / Qwen)    │  ← Pre-trained foundation
│  • Understands C#, English, logic       │
│  • Open-source, free to use             │
└─────────────────────────────────────────┘
```

You brand it "cdpdevopsphenox" — it's your product name. The base model is like an engine in a car; you don't manufacture the engine, you build around it.

### Three Ways to Add Knowledge

| Method | Effort | Result |
|--------|--------|--------|
| **RAG** (retrieval) | Low | Model searches your code at query time |
| **System Prompt** | Low | Model follows your rules/patterns |
| **Fine-tuning** | Medium | Model internalizes your patterns permanently |

### Why Hardware Matters

LLMs are massive matrix multiplications. Each parameter ≈ 0.5–1 byte in memory:

| Model Size | Parameters | RAM Needed |
|-----------|------------|------------|
| 7B | 7 billion | ~4-8 GB |
| 16B | 16 billion | ~10-16 GB |
| 70B | 70 billion | ~40-48 GB |

GPUs do ~10,000 parallel operations vs ~10 on CPU — that's the speed difference.

| Hardware | Model | Speed |
|----------|-------|-------|
| Regular laptop (16GB, no GPU) | 7B (CPU) | ~5-15 sec/response |
| Laptop with 8GB GPU | 7B (GPU) | ~1-3 sec/response |
| Workstation with 16GB GPU | 16B (GPU) | ~1-3 sec/response |

A 7B quantized model on CPU is usable, free, and fully private.

---

## Next Steps (To Discuss)

- [x] ~~Choose approach: RAG pipeline vs Continue.dev vs AnythingLLM~~ → **RAG pipeline chosen**
- [x] ~~Choose UI: Web (Gradio) vs VS Code Chat~~ → **VS Code Chat Participant (`@cdpdevopsphenox`)**
- [ ] Decide on base model: DeepSeek Coder v2 (16B) vs Qwen 2.5 Coder (7B/14B)
- [ ] Define fine-tuning dataset: curate Q&A pairs from team knowledge
- [ ] Build prototype and validate with team
- [ ] Plan distribution: how developers install and use "cdpdevopsphenox"

## Part 4: VS Code Chat Integration Decision

Developer chose VS Code native chat over web UI (Gradio). The architecture was updated:

- **Backend:** Python FastAPI server (`server.py`) on `localhost:8321` exposes RAG pipeline as REST API
- **Frontend:** VS Code extension registers a Chat Participant (`@cdpdevopsphenox`)
- **Commands:** `/flow`, `/pattern`, `/package`, `/reset`
- **Features:** Conversation memory, source file citations, health checks, streaming support

Build plan updated at `C:\gitrnd\cdpdevopsphenox-build-plan.md`.

---

*Session continues...*
