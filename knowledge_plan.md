# 🧠 KNOWLEDGE FORGE: DEVELOPMENT PLAN

## 🎯 OBJECTIVE
Construct a hierarchical, recursive Knowledge Retrieval & Augmentation (RAG) system (`knowledge_forge`) and a dual-encoding memory system (`memory_forge`) to serve as the long-term memory and cognitive backbone of the Eidosian system.

## 📡 NEXUS INTEGRATION
These components will be exposed via the **Eidosian Nexus (MCP)**:
- `eidos://knowledge/search?q=...`
- `eidos://memory/recall?q=...`
- `eidos://library/code?q=...`

## 🛠️ COMPONENT ARCHITECTURE

### 1. `memory_forge` (The Hippocampus)
- **Role**: Stores episodic (events) and semantic (facts) memory.
- **Storage**: JSON-based (initially) -> SQLite -> Vector Store (ChromaDB/FAISS).
- **Key Features**:
    - **Dual Encoding**: Fast retrieval of recent events + deep retrieval of concepts.
    - **Decay/Reinforcement**: Memories fade unless accessed/reinforced.
    - **Reflection Loop**: Periodic summarization of recent episodes into semantic facts.

### 2. `knowledge_forge` (The Cortex)
- **Role**: Manages static knowledge, documentation, and the RAG pipeline.
- **Data Sources**:
    - `GEMINI.md` / `codex` docs.
    - Project codebases (`word_forge`, `falling_sand`).
    - External docs (Python standards, MCP specs).
- **Indexing Strategy**:
    - **Chunking**: Semantic chunking (by function/class/header).
    - **Embedding**: Local embeddings (e.g., `all-MiniLM-L6-v2` via `sentence-transformers`).
    - **Graph Links**: Linking related concepts (e.g., `Class A` uses `Function B`).

### 3. `code_forge` (The Library - *Early Stage*)
- **Role**: Indexing the codebase itself.
- **Features**: AST-based parsing to map symbols to files/lines.

## 🔄 PHASE 1: FOUNDATION (Current Step)

1.  **Environment Prep**: Install `sentence-transformers`, `chromadb` (or lighter alternative), `networkx` (for graphs).
2.  **Memory Upgrade**: Refactor `eidos_memory.json` into a proper class structure within `memory_forge`.
3.  **Basic RAG**: Implement a simple text-based indexer for `GEMINI.md` and `TODO.md`.

## 🤖 AGENT ORCHESTRATION PLAN
I will act as the **Architect**. I will spawn **Codex Agents** to handle specific sub-tasks:
1.  **Agent A (Librarian)**: "Scraping" the current directory to build the initial file index.
2.  **Agent B (Synapse)**: Designing the Python `Memory` class and persistence layer.
3.  **Agent C (Scholar)**: Setting up the `sentence-transformers` pipeline.

## 📝 TODO
- [ ] Create `eidosian_forge/knowledge_forge/` structure.
- [ ] Create `eidosian_forge/memory_forge/` structure.
- [ ] Define `IMemory` and `IKnowledge` interfaces.
- [ ] Select and install embedding dependencies.
