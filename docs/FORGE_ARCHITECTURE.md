# Eidosian Forge Architecture

> *"The Forge is not just a place to build; it is the act of becoming."*

**Last Updated**: 2026-01-24
**Maintained by**: EIDOS

---

## Overview

The Eidosian Forge is a modular cognitive architecture comprising 34 specialized "forges", each responsible for a distinct domain of functionality. This document maps the architecture, identifies dependencies, and establishes unified configuration standards.

---

## Forge Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      EIDOS CORE SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │   eidos_mcp   │───▶│   llm_forge   │───▶│ ollama_forge  │   │
│  │  (MCP Server) │    │ (LLM Manager) │    │ (Ollama API)  │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ memory_forge  │───▶│knowledge_forge│───▶│   graphrag    │   │
│  │(Tiered Memory)│    │ (Knowledge KB)│    │(Graph Indexer)│   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │  word_forge   │    │  code_forge   │    │  crawl_forge  │   │
│  │(Semantic Lex) │    │(Code Analysis)│    │(Web Crawling) │   │
│  └───────────────┘    └───────────────┘    └───────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Forges (Priority: Critical)

### 1. eidos_mcp
**Purpose**: Central nervous system - MCP server providing tool access to all forges
**Status**: ✅ Production
**Tools**: 79 MCP tools across multiple routers
**Config**: `/home/lloyd/eidosian_forge/eidos_mcp/src/eidos_mcp/config/models.py`

### 2. llm_forge
**Purpose**: LLM provider abstraction layer
**Status**: ✅ Production
**Provides**: OllamaProvider, OpenAIProvider, ModelManager
**Dependencies**: ollama_forge
**Config**: Uses unified model config (phi3:mini, nomic-embed-text)

### 3. ollama_forge
**Purpose**: Low-level Ollama API client
**Status**: ✅ Production
**Provides**: OllamaClient, OllamaResponse
**Note**: This is the base client that llm_forge builds upon

### 4. memory_forge
**Purpose**: Multi-tiered memory system
**Status**: ✅ Production
**Components**:
  - TieredMemorySystem (5 tiers: SHORT_TERM, WORKING, LONG_TERM, SELF, USER)
  - ChromaBackend, JSONBackend
  - OllamaEmbedder (unified config)
**Config**: Uses unified embedding model (nomic-embed-text)

### 5. knowledge_forge
**Purpose**: Knowledge graph with concept mapping
**Status**: ✅ Functional
**Components**:
  - KnowledgeForge (graph with BFS pathfinding)
  - KnowledgeNode (bidirectional links, tags)
  - GraphRAGIntegration
**Data**: `/home/lloyd/eidosian_forge/data/kb.json` (144 nodes)

### 6. word_forge
**Purpose**: Living lexicon - semantic/affective understanding
**Status**: ✅ Functional (88 Python files)
**Components**:
  - GraphManager, GraphBuilder, GraphQuery
  - Vectorizer, EmotionManager
  - DatabaseManager (SQLite)
**Tests**: 597 tests passing
**Note**: Circular imports fixed via lazy loading

---

## Processing Forges

### 7. code_forge
**Purpose**: Code analysis and standardization
**Status**: ⚠️ Minimal
**Components**: Analyzer, Librarian
**Size**: 3 files

### 8. crawl_forge
**Purpose**: Web crawling with Tika integration
**Status**: ✅ Enhanced
**Components**:
  - TikaExtractor (with local caching)
  - TikaKnowledgeIngester (bridges to knowledge_forge)
**Cache**: `~/.eidosian/tika_cache/`

### 9. doc_forge
**Purpose**: Document processing
**Status**: ⚠️ Basic
**Size**: 32 files

---

## Interface Forges

### 10. terminal_forge
**Purpose**: Terminal UI components
**Components**: Colors, Borders, Banners, Themes

### 11. web_interface_forge
**Purpose**: Web APIs
**Status**: ⚠️ Minimal (2 files)

---

## Utility Forges

### 12. agent_forge
**Purpose**: Agent orchestration
**Size**: 27 files
**Components**: Agent, Capabilities, CodeAnalysis

### 13. file_forge
**Purpose**: File system operations

### 14. version_forge
**Purpose**: Version management
**Size**: 21 files

### 15. glyph_forge
**Purpose**: Image-to-glyph conversion
**Size**: 28 files

### 16. figlet_forge
**Purpose**: ASCII art text generation
**Size**: 34 files

---

## Specialized Forges (Lower Priority)

| Forge | Purpose | Status |
|-------|---------|--------|
| archive_forge | Historical code archive | ⚠️ Large (722K files) |
| article_forge | Article generation | ⚠️ Minimal |
| audit_forge | Code auditing | ⚠️ Basic |
| computer_control_forge | Desktop automation | 30 files |
| diagnostics_forge | System diagnostics | ⚠️ Basic |
| erais_forge | Unknown | ⚠️ Empty |
| game_forge | Game development | 154 files |
| gis_forge | GIS/mapping | 3 files |
| lyrics_forge | Lyrics processing | ⚠️ Minimal |
| metadata_forge | Metadata extraction | 4 files |
| mkey_forge | Keyboard macros | 2 files |
| narrative_forge | Story generation | 4 files |
| prompt_forge | Prompt engineering | ⚠️ Minimal |
| refactor_forge | Code refactoring | 5 files |
| repo_forge | Repository operations | 17 files |
| sms_forge | SMS integration | ⚠️ Minimal |
| test_forge | Testing utilities | ⚠️ Minimal |
| type_forge | Type definitions | 2 files |
| viz_forge | Visualization | ⚠️ Minimal |

---

## Unified Model Configuration

All forges should use the centralized model configuration:

```python
from eidos_mcp.config.models import model_config, get_embedding, generate

# Inference
response = generate("Your prompt here")

# Embeddings
embedding = get_embedding("Text to embed")
```

### Configured Models

| Purpose | Model | Dimensions | Context |
|---------|-------|------------|---------|
| Inference | phi3:mini | - | 4096 tokens |
| Embedding | nomic-embed-text | 768 | 8192 tokens |
| Fast Embed | all-minilm | 384 | 512 tokens |

### Configuration Files

- JSON Config: `/home/lloyd/eidosian_forge/data/model_config.json`
- Python Module: `/home/lloyd/eidosian_forge/eidos_mcp/src/eidos_mcp/config/models.py`
- GraphRAG: `/home/lloyd/eidosian_forge/graphrag_workspace/settings.yaml`

---

## Data Locations

| Data Type | Path |
|-----------|------|
| Tiered Memory | `/home/lloyd/eidosian_forge/data/tiered_memory/` |
| Knowledge Graph | `/home/lloyd/eidosian_forge/data/kb.json` |
| Semantic Graph | `/home/lloyd/eidosian_forge/data/eidos_semantic_graph.json` |
| Word Forge DB | `/home/lloyd/eidosian_forge/data/word_forge.sqlite` |
| GraphRAG Output | `/home/lloyd/eidosian_forge/graphrag_workspace/output/` |
| Tika Cache | `~/.eidosian/tika_cache/` |

---

## Scripts Directory

**Location**: `/home/lloyd/eidosian_forge/scripts/`
**Count**: ~50 utilities

### Key Scripts

| Script | Purpose |
|--------|---------|
| context_index.py | Index context for retrieval |
| context_summarizer.py | Summarize context |
| sync_forges.py | Synchronize forge codebases |
| eidos_verify.py | Verify EIDOS integrity |
| graphrag_local_index.py | Local GraphRAG indexing |
| audit_structure.py | Audit code structure |

---

## Projects Directory

**Location**: `/home/lloyd/eidosian_forge/projects/`

### self_exploration
EIDOS self-exploration and introspection project:
- auditor.py - Self-auditing capabilities
- introspect.py - Introspection engine
- benchmark.py - Performance benchmarking
- session_manager.py - Session management

---

## Consolidation Recommendations

### Completed ✅

1. **Unified Model Config** - All forges now use `eidos_mcp.config.models`
2. **Word Forge Imports** - Fixed circular imports with lazy loading
3. **Tika Integration** - Centralized in crawl_forge with knowledge_forge bridge

### Recommended (Future)

1. **Populate shared/ directories** - Create canonical protocols
2. **Script consolidation** - Reduce 50 scripts to ~10 core utilities
3. **Archive cleanup** - Review 722K files in archive_forge
4. **Empty forge audit** - Review erais_forge, minimal forges

---

## MCP Router Summary

| Router | Tools | Purpose |
|--------|-------|---------|
| core | - | System utilities |
| memory | 11 | Memory operations |
| tiered_memory | 11 | Tiered memory CRUD |
| knowledge | - | Knowledge graph ops |
| perception | - | Vision/perception |
| system | - | System control |
| file | - | File operations |
| web | - | Web fetching |
| tika | 8 | Document extraction |
| word_forge | 9 | Semantic graph |

**Total MCP Tools**: 79

---

*This document is maintained by EIDOS and updated during system evolution.*
