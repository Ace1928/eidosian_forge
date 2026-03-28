# 📖 Word Forge ⚡

> _"The Lexicon of Eidos. Structuring the semantic fabric of intelligence."_

## 🧠 Overview

`word_forge` is the multidimensional semantic engine for the Eidosian ecosystem. It builds and maintains a comprehensive semantic graph, providing deep lexical analysis, emotion tracking (valence/arousal), and vector-backed similarity search. It serves as the primary linguistic source of truth, ensuring that agents share a common, highly structured vocabulary.

```ascii
      ╭───────────────────────────────────────────╮
      │               WORD FORGE                  │
      │    < Lexicon | Emotion | Graph | Search > │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   SEMANTIC NETWORK  │   │ EMOTION ENGINE  │
      │ (NetworkX / SQLite) │   │ (VADER / TB)    │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Lexical Processing & Semantic Mapping
- **Test Coverage**: 570+ tests passing (High-fidelity logic verification).
- **MCP Integration**: 9 Tools (`wf_add_term`, `wf_find_related`, `wf_graph_stats`, etc.).
- **Architecture**:
  - `graph/`: NetworkX-based knowledge graph construction and visualization.
  - `database/`: SQLite persistence for terms and semantic relationships.
  - `emotion/`: Multi-modal emotion analysis using heuristic and LLM-backed strategies.
  - `queue/`: Non-blocking background workers for graph enrichment and vector indexing.

## 🚀 Usage & Workflows

### Local LLM Backend

- **Default local backend**: `gguf:models/Qwen3.5-0.8B-Q4_K_M.gguf` when present.
- **Fallback backend**: `qwen/qwen3.5-2b-instruct` when no local GGUF is available.
- **Runtime override**: set `EIDOS_WORD_FORGE_LLM_MODEL` to an Ollama model, Hugging Face model name, or `gguf:/path/to/model.gguf`.
- **llama.cpp override**: set `EIDOS_WORD_FORGE_LLAMA_CLI` if `llama-cli` lives outside the standard forge paths.

Word Forge now prefers the smallest practical local Qwen 3.5 GGUF for low-latency structured lexical tasks, while keeping heavier model paths available as explicit overrides.

### Lexical Ingestion (CLI)

Start the autonomous processing pipeline with seed terms:
```bash
word_forge start intelligence consciousness --minutes 10 --workers 4
```

### LLM Backend Benchmark

```bash
./eidosian_venv/bin/python word_forge/scripts/benchmark_llm_backends.py \
  --model gguf:models/Qwen3.5-0.8B-Q4_K_M.gguf
```

### Semantic Graph (Python)

```python
from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager

db = DBManager()
graph = GraphManager(db_manager=db)

# Build and visualize the network
graph.build_graph()
graph.visualize(output_path="reports/word_graph.html")
```

### Emotion Analysis

```python
from word_forge.emotion.emotion_manager import EmotionManager

em = EmotionManager(db_manager=db)
valence, arousal = em.analyze_text_emotion("The Eidosian future is vibrant and precise.")
print(f"Valence: {valence}, Arousal: {arousal}")
```

## 🔗 System Integration

- **Knowledge Forge**: Provides the low-level lexical terms that form the basis of high-level knowledge nodes.
- **Narrative Forge**: Uses emotion analysis to maintain thematic tone and character consistency.
- **Prompt Forge**: Injects precise semantic synonyms into templates to optimize model alignment.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Stabilize multi-threaded background workers in Termux.

### Future Vector (Phase 3+)
- Implement "Cross-Lingual Synapse" where terms are automatically mapped across different language substrates in real-time.
- Deep integration with `llm_forge` for autonomous "Concept Discovery" sweeps.

---
*Generated and maintained by Eidos.*
