# Word Forge Comprehensive TODO

> **Unified Improvement Roadmap**  
> **Last Updated**: 2026-01-12  
> **Status**: Active Development
>
> This document consolidates ALL improvement opportunities, architectural enhancements, and planned features for Word Forge. It merges content from:
> - `TODO.md` (MVP deliverables)
> - `upgrade_plan.md` (architectural review)
> - `code_perfection_prompt.md` (Eidosian transformation patterns)
> - `code_polishing_prompt.md` (refinement directives)
> - `living_lexicon_discussion.md` (original vision)
> - Exhaustive file-by-file analysis

---

## Format Guide

```
Legend:
  [ ] = Not started
  [~] = In progress  
  [x] = Complete
  [!] = Blocked/Needs discussion

Priority Levels:
  P0 = Critical (blocking MVP)
  P1 = High (needed soon)
  P2 = Medium (planned)
  P3 = Low (nice to have)

Item Format:
  - [ ] **`file/path.py:line`** - Brief description [Priority]
    - Impact: Why this matters
    - Implementation: How to fix/implement
    - Dependencies: What this depends on
```

---

## Table of Contents

1. [MVP Deliverables](#1-mvp-deliverables)
2. [Code Quality](#2-code-quality)
3. [Type Safety](#3-type-safety)
4. [Documentation](#4-documentation)
5. [Testing](#5-testing)
6. [Configuration](#6-configuration)
7. [Database](#7-database)
8. [Graph Module](#8-graph-module)
9. [Emotion Module](#9-emotion-module)
10. [Vector Module](#10-vector-module)
11. [Queue & Workers](#11-queue--workers)
12. [Parser Module](#12-parser-module)
13. [Conversation Module](#13-conversation-module)
14. [CLI](#14-cli)
15. [Error Handling](#15-error-handling)
16. [Performance](#16-performance)
17. [Security](#17-security)
18. [Architecture](#18-architecture)
19. [CI/CD](#19-cicd)
20. [Dependencies](#20-dependencies)
21. [Future Vision](#21-future-vision)
22. [Eidosian Checklist](#22-eidosian-checklist)

---

## 1. MVP Deliverables

> Source: `TODO.md` - Core deliverables for initial release

### 1.1 Test Coverage [P0]

- [x] **`tests/test_queue_manager.py`** - Queue manager tests
- [x] **`tests/test_database_worker.py`** - Database worker tests [P0]
  - Status: Complete - 59 tests covering state, exceptions, metrics, lifecycle
- [x] **`tests/test_emotion_processor.py`** - Emotion processor tests [P0]
  - Status: Complete - 40 tests covering initialization, context, hooks, relationships
- [x] **`tests/test_conversation_worker.py`** - Conversation worker tests [P1]
  - Status: Complete - 33 tests covering state, metrics, exceptions
- [x] **`tests/test_graph_builder.py`** - Graph builder tests [P1]
  - Status: Complete - 12 tests covering initialization, build, update, verification
- [x] **`tests/test_graph_io.py`** - Graph I/O tests [P1]
  - Status: Complete - 13 tests covering GEXF save/load, subgraph export
- [x] **`tests/test_graph_query.py`** - Graph query tests [P1]
  - Status: Complete - 23 tests covering node/edge queries, subgraph extraction
- [x] **`tests/test_language_model.py`** - LLM interface tests [P1]
  - Status: Complete - 23 tests covering initialization, generation, error handling
- [x] **Real dependency coverage** - Remove torch/chromadb/transformers stubs [P0]
  - Impact: Integration tests validate actual ingestion, vector, and LLM flows
  - Implementation: Replace mock fixtures with real instances; use small models in tests
  - Status: Complete - Tests now run with real dependencies

### 1.2 Vectorization [P0]

- [ ] **Verify persistence** - Embeddings survive restart
- [ ] **Polling logic** - Detect new/updated entries
- [x] **CLI search command** - `word_forge vector search <query>`
  - Status: Complete - Added search subcommand with query, top-k, and content-type options

### 1.3 Graph Module [P0]

- [x] **Rebuild from database** - `GraphManager.build_graph()` works
- [ ] **Incremental updates** - Avoid full rebuilds
- [x] **Export utilities** - GEXF, GraphML formats
  - Status: Complete - GraphIO supports save_to_gexf, load_from_gexf, and export_subgraph

### 1.4 Conversation & Emotion [P1]

- [x] **Store conversations** - Multi-turn persistence
  - Status: Complete - ConversationManager stores conversations in SQLite
- [ ] **Emotion integration** - Auto-annotate messages
- [x] **CLI commands** - `word_forge conversation start/list`
  - Status: Complete - Added start, list, and show subcommands

### 1.5 Worker Orchestration [P1]

- [x] **WorkerManager** - Lifecycle management complete
- [ ] **Enable/disable flags** - Per-worker configuration

---

## 2. Code Quality

### 2.1 Docstrings [P1]

- [x] **`src/word_forge/database/database_manager.py:1-64`** - Malformed docstring
  - Impact: Docstring content appears before description
  - Fix: Move description to top, sections below
  - Status: Complete - Docstring reorganized with proper structure
- [x] **`src/word_forge/parser/parser_refiner.py`** - Missing module docstring
  - Status: Complete - Added comprehensive module docstring
- [x] **`src/word_forge/conversation/conversation_manager.py`** - Missing module docstring
  - Status: Complete - Added comprehensive module docstring
- [x] **`src/word_forge/demos/*.py`** - Demo files lack docstrings
  - Status: Complete - Added docstrings to __init__.py, cli_demo.py, tools_demo.py
- [x] **`src/word_forge/demos/conversation_worker_demo.py:202`** - Invalid Result API usage
  - Status: Complete - Replaced `is_ok()` with `is_success`

### 2.2 File Organization [P2]

- [ ] **`src/word_forge/config.py`** - Split 1285-line file
  - Target: `config_loader.py`, `config_observers.py`, `config_profiles.py`
- [ ] **`src/word_forge/configs/config_essentials.py`** - Split 1697-line file
  - Target: `types.py`, `errors.py`, `protocols.py`, `utilities.py`
- [ ] **`src/word_forge/vectorizer/vector_store.py`** - Split 1757-line file
  - Target: `vector_backends.py`, `vector_search.py`, `vector_metadata.py`

### 2.3 Constants [P2]

- [x] **`src/word_forge/forge.py:121`** - Extract `0.5` to `MAIN_LOOP_SLEEP_INTERVAL`
  - Status: Complete - Added constant at module level
- [x] **`src/word_forge/forge.py:122`** - Extract `5` to `PROGRESS_REPORT_INTERVAL`
  - Status: Complete - Added constant at module level
- [ ] **`src/word_forge/emotion/emotion_manager.py:177`** - Extract `100` to config
- [ ] **`src/word_forge/parser/parser_refiner.py:93-117`** - Move stop words to config

### 2.4 Dead Code [P3]

- [ ] **`src/word_forge/exceptions.py:126-182`** - Remove empty comment sections
- [ ] **Duplicate exceptions** - Consolidate `DatabaseError` locations

---

## 3. Type Safety

### 3.1 Remove `Any` Usage [P1]

- [ ] **`src/word_forge/config.py`** - Replace `Any` with specific types
- [ ] **`src/word_forge/queue/queue_manager.py:23`** - `ErrorContext.context`
- [ ] **`src/word_forge/vectorizer/vector_store.py`** - Reduce `cast` usage

### 3.2 Add Missing Hints [P2]

- [ ] **`src/word_forge/parser/parser_refiner.py`** - `extract_terms` return type
- [ ] **`src/word_forge/conversation/conversation_types.py`** - Protocol methods

### 3.3 Fix Type Ignores [P2]

- [ ] **`src/word_forge/parser/parser_refiner.py:9-14`** - NLTK stubs
- [ ] **`src/word_forge/vectorizer/vector_store.py:48-65`** - Optional import stubs

---

## 4. Documentation

### 4.1 README [P1]

- [x] **Architecture diagram** - Added
- [x] **Troubleshooting section** - NLTK, models, memory issues
  - Status: Complete - Added comprehensive troubleshooting for NLTK, memory, ChromaDB, SQLite, and tests
- [ ] **Performance benchmarks** - Data size capabilities
- [x] **API quick reference** - Main classes table
  - Status: Complete - Added Core Classes table and Common Operations examples

### 4.2 Module Docs [P2]

- [ ] **`docs/adr/`** - Architecture Decision Records
- [ ] **API reference** - Sphinx/MkDocs generation
- [ ] **`docs/overview.md`** - Add sequence diagrams
- [ ] **`docs/examples/`** - Usage examples directory

### 4.3 Inline Docs [P2]

- [ ] **`src/word_forge/graph/graph_analysis.py`** - Algorithm explanations
- [ ] **`src/word_forge/emotion/emotion_processor.py`** - Recursive processing docs

---

## 5. Testing

### 5.1 Missing Tests [P0-P1]

| Test File | Status | Priority |
|-----------|--------|----------|
| `test_database_worker.py` | ✅ Complete | P0 |
| `test_emotion_processor.py` | ✅ Complete | P0 |
| `test_conversation_worker.py` | ✅ Complete (33 tests) | P1 |
| `test_graph_builder.py` | ✅ Complete (12 tests) | P1 |
| `test_graph_io.py` | ✅ Complete (13 tests) | P1 |
| `test_graph_query.py` | ✅ Complete (23 tests) | P1 |
| `test_language_model.py` | ✅ Complete (23 tests) | P1 |
| `test_parser_config.py` | ✅ Complete (17 tests) | P2 |
| `test_ingestion_pipeline.py` | ✅ Complete (end-to-end ingestion) | P0 |

### 5.2 Test Infrastructure [P1]

- [x] **`tests/conftest.py`** - Shared fixtures (db, config)
  - Status: Complete - Added db + graph fixtures and sample data helpers (no mocks)
- [x] **`tests/fixtures/`** - Sample data files
  - Status: Complete - Added sample_words.json, sample_conversations.json, sample_thesaurus.jsonl
- [x] **`tests/test_lexical_functions.py`** - Verify lexical I/O helpers (lexical proto removed)
  - Status: Complete - Uses real helpers to avoid dependency stubs or legacy scripts
- [ ] **`pyproject.toml`** - Coverage thresholds
- [ ] **CI cache for model/NLTK downloads** - Reduce runtime for integration tests [P1]

### 5.3 Test Quality [P2]

- [ ] **Property-based tests** - Hypothesis for data transforms
- [ ] **Performance tests** - pytest-benchmark for critical paths
- [ ] **Snapshot tests** - Graph visualization outputs

---

## 6. Configuration

### 6.1 Validation [P1]

- [ ] **Cross-component validation** - `validate_all()` improvements
- [ ] **Path validation** - Writability checks
- [ ] **Model name validation** - Against known models

### 6.2 Environment [P2]

- [ ] **Complete ENV_VARS** - All settings overridable
- [ ] **`.env.example`** - Document all variables

### 6.3 Profiles [P2]

- [ ] **External profiles** - Load from YAML/JSON
- [ ] **Minimal profile** - Disable heavy features
- [ ] **Hot reload propagation** - Update components on change

### 6.4 Runtime Cache [P1]

- [ ] **`src/word_forge/config.py:772`** - Invalidate `get_cached_value` LRU entries when config changes [P1]
  - Impact: `set_runtime_value`/env updates can return stale values from `get_cached_value`.
  - Implementation: remove unused `_value_cache` or add explicit `get_cached_value.cache_clear()` (or per-key invalidation) when mutating values.

---

## 7. Database

### 7.1 Schema [P1]

- [ ] **Migrations** - Alembic-style system
- [ ] **Version tracking** - `schema_version` table
- [ ] **Foreign keys** - Enable `PRAGMA foreign_keys`

### 7.2 Performance [P2]

- [ ] **Query analysis** - EXPLAIN for slow queries
- [ ] **Indexes** - Frequent query columns
- [ ] **Prepared statements** - Statement cache
- [ ] **`src/word_forge/database/database_manager.py:543`** - Avoid returning pooled connections still stored in `_conn_pool` [P1]
  - Impact: `create_connection()` appends to the pool and returns the same connection, allowing concurrent reuse and pool corruption.
  - Implementation: only add connections to the pool after callers release them (or remove `create_connection` from pool management entirely).

### 7.3 Abstraction [P2]

- [ ] **Remove direct sqlite3** - Route through DBManager
  - `conversation_manager.py` - Uses direct sqlite3
  - `vector_store.py` - Some direct calls

---

## 8. Graph Module

### 8.1 Construction [P1]

- [ ] **Incremental updates** - Delta processing
- [ ] **Graph versioning** - Track changes over time

### 8.2 Analysis [P2]

- [ ] **More algorithms** - Leiden, Infomap
- [ ] **Graph embeddings** - node2vec, GraphSAGE
- [ ] **Path analysis** - Semantic path finding

### 8.3 Visualization [P2]

- [ ] **More layouts** - Fruchterman-Reingold, Kamada-Kawai
- [ ] **Interactive filtering** - D3.js integration
- [ ] **Legend** - Relationship type key

### 8.4 I/O [P2]

- [ ] **More formats** - Cypher, DOT, JSON-LD
- [ ] **Streaming export** - Large graph handling

---

## 9. Emotion Module

### 9.1 Analysis [P2]

- [ ] **Configurable weights** - VADER/TextBlob per domain
- [ ] **Trend analysis** - Temporal patterns
- [ ] **Configurable depth** - Recursive analysis limits

### 9.2 Models [P2]

- [ ] **More taxonomies** - Plutchik wheel, PAD model
- [ ] **Composite emotions** - Combination logic

### 9.3 Persistence [P2]

- [ ] **History tracking** - Emotion timeline
- [ ] **Confidence decay** - Time-based weighting

---

## 10. Vector Module

### 10.1 Storage [P2]

- [ ] **More backends** - Pinecone, Weaviate, Milvus
- [ ] **Version tracking** - Model version per embedding
- [ ] **Compression** - PQ/OPQ support

### 10.2 Models [P2]

- [ ] **Dynamic model selection** - Any sentence-transformers
- [ ] **Quality metrics** - Embedding benchmarks

### 10.3 Search [P2]

- [ ] **Hybrid search** - BM25 + vector
- [ ] **Result explanation** - Similarity breakdown

---

## 11. Queue & Workers

### 11.1 Queue [P2]

- [ ] **Persistence** - SQLite/Redis queue
- [ ] **Dead letter queue** - Failed items
- [ ] **Rate limiting** - Token bucket
- [ ] **`src/word_forge/queue/queue_manager.py:370`** - Return a snapshot for iteration instead of a live iterator [P2]
  - Impact: `__iter__` exposes the internal queue list without holding the lock, risking race conditions and inconsistent iteration.
  - Implementation: copy `list(self._queue.queue)` under lock and iterate over the snapshot.

### 11.2 Workers [P2]

- [ ] **Health checks** - Heartbeat monitoring
- [ ] **Auto-restart** - Failure recovery
- [ ] **Auto-scaling** - Queue depth based

### 11.3 Coordination [P3]

- [ ] **Distributed locking** - Multi-process safety
- [ ] **Work stealing** - Load balancing

---

## 12. Parser Module

### 12.1 Extraction [P2]

- [ ] **SpaCy pipeline** - NER, dependencies
- [ ] **MWE detection** - Multi-word expressions
- [ ] **Domain dictionaries** - Technical terms
- [x] **`src/word_forge/parser/parser_refiner.py:490`** - Respect provided queue manager even when empty
  - Status: Complete - Switched to explicit None check to avoid replacing empty queues

### 12.2 Resources [P2]

- [ ] **Lazy loading** - On-demand resources
- [ ] **Update mechanism** - Resource versioning

### 12.3 LLM [P2]

- [ ] **Remove ModelState singleton** - Dependency injection
- [ ] **Configurable models** - Move names to config
- [ ] **Memory management** - Model unloading
- [x] **`src/word_forge/parser/language_model.py:80`** - Avoid accelerate requirement on CPU
  - Status: Complete - Use device_map only when accelerate is available

---

## 13. Conversation Module

### 13.1 Management [P2]

- [ ] **Full-text search** - Message search
- [ ] **Summarization** - LLM-based summaries
- [ ] **Branching** - Conversation trees
- [x] **LLM-backed model implementations** - Replace mock-only conversation models [P1]
  - Status: Complete - Added LLM-backed reflexive/lightweight/affective models and updated demos/tests
- [ ] **`src/word_forge/conversation/conversation_manager.py:237`** - Fail fast if table creation fails during init [P1]
  - Impact: initialization errors are printed then ignored, leaving a partially usable manager.
  - Implementation: log and re-raise `ConversationError` so callers can handle startup failures explicitly.
- [ ] **`src/word_forge/conversation/conversation_manager.py:151`** - Normalize DBManager exceptions into ConversationError [P2]
  - Impact: `_db_connection` surfaces `DatabaseError` but callers only catch `ConversationError`, reducing error clarity.
  - Implementation: catch `DatabaseError` in `_db_connection` and wrap/raise `ConversationError` with context.

### 13.2 Messages [P3]

- [ ] **Threading** - Reply relationships
- [ ] **Attachments** - Multimedia support

### 13.3 Analysis [P3]

- [ ] **Topic modeling** - Topic extraction
- [ ] **Speaker analysis** - Per-speaker stats

---

## 14. CLI

### 14.1 Commands [P1]

- [x] **`--version`** - Version display (added `--version` and `-V` flags)
- [x] **`--config`** - Config file option (added `--config`/`-c` flag)
- [x] **`--quiet`** - Suppress output (added `--quiet`/`-q` and `--verbose`/`-v` flags)
- [x] **`--llm-model`** - Override example generation model

### 14.2 UX [P2]

- [ ] **Progress bars** - tqdm/rich
- [ ] **Colored output** - rich/colorama
- [ ] **Shell completion** - bash/zsh scripts

### 14.3 Organization [P3]

- [ ] **Plugin system** - Extensible commands
- [ ] **Aliases** - Short command forms

---

## 15. Error Handling

### 15.1 Hierarchy [P1]

- [x] **VectorError** - Vector operation errors
  - Status: Complete - Added VectorError, VectorStorageError, VectorSearchError, VectorIndexError, VectorEmbeddingError
- [x] **ConversationError** - In central exceptions.py
  - Status: Complete - Added ConversationError, ConversationNotFoundError, ConversationStateError
- [x] **Error codes** - Unique identifiers
  - Status: Complete - Added ErrorCode class with WF-{CATEGORY}-{NUMBER} format codes

### 15.2 Result Pattern [P2]

- [ ] **Consistent usage** - All modules
- [ ] **Combinators** - `and_then`, `or_else`

---

## 16. Performance

### 16.1 Memory [P2]

- [ ] **Model unloading** - Unused model cleanup
- [ ] **Graph sharding** - Large graph handling
- [ ] **LRU cache limits** - Bounded caches

### 16.2 CPU [P2]

- [ ] **Batch processing** - Term extraction
- [ ] **Batch emotion** - Analysis batching

### 16.3 I/O [P2]

- [ ] **Pool metrics** - Connection utilization
- [ ] **Async embeddings** - Non-blocking generation

### 16.4 Profiling [P3]

- [ ] **cProfile integration** - Built-in profiling
- [ ] **Prometheus metrics** - Observability

---

## 17. Security

### 17.1 Input [P1]

- [ ] **SQL injection audit** - Parameterized queries
- [ ] **Input sanitization** - Parser inputs

### 17.2 Data [P2]

- [ ] **Secret masking** - Log/export protection
- [ ] **Encryption** - SQLCipher support

### 17.3 Dependencies [P1]

- [ ] **Version constraints** - Upper bounds
- [ ] **Vulnerability scanning** - Dependabot/Snyk

---

## 18. Architecture

> Source: `upgrade_plan.md` - Architectural patterns

### 18.1 Result Pattern [P1]

- [ ] **Standardize error handling** - Return `Result[T]` instead of exceptions
- [ ] **Error context codes** - Consistent error identification
  - Impact: Explicit error paths, better composability
  - Breaking: Requires coordinated updates

### 18.2 Repository Pattern [P2]

- [ ] **Define protocols** - `WordRepository`, `ConversationRepository`
- [ ] **SQLite implementations** - `SQLiteWordRepository`, etc.
- [ ] **Dependency injection** - Accept protocols in constructors
  - Impact: Decouples from SQLite, enables mocking
  - Breaking: Component initialization changes

### 18.3 Configuration Refactor [P2]

- [ ] **Simple loader** - `load_config() -> Dict`
- [ ] **Component injection** - Pass config via constructors
- [ ] **Remove global config** - No `config.database.db_path`
  - Impact: Better testability, explicit dependencies
  - Breaking: Fundamental access pattern change

### 18.4 LLM Abstraction [P2]

- [ ] **Define `LLMInterface` protocol** - `generate()`, `embed()`
- [ ] **HuggingFace implementation** - Encapsulate transformers
- [ ] **Remove ModelState singleton** - Use dependency injection
  - Impact: Swap models easily, testable
  - Breaking: Singleton removal

### 18.5 Event System [P3]

- [ ] **Event bus** - Cross-component communication
- [ ] **Plugin architecture** - Extensible analyzers/backends

---

## 19. CI/CD

### 19.1 Pipeline [P1]

- [x] **Dependency caching** - pip cache
- [ ] **Parallel tests** - pytest-xdist
- [ ] **Coverage reporting** - codecov/coveralls
- [ ] **Integration test job** - Separate workflow

### 19.2 Release [P2]

- [ ] **Release workflow** - semantic-release
- [ ] **Changelog** - conventional-changelog

### 19.3 Containers [P2]

- [ ] **Dockerfile** - Multi-stage build
- [ ] **docker-compose.yml** - Dev environment

---

## 20. Dependencies

### 20.1 Management [P1]

- [ ] **Upper bounds** - Prevent breaking changes
- [ ] **Lock file** - pip-tools or Poetry
- [x] **TextBlob dependency** - Ensure emotion analysis matches requirements
  - Status: Complete - Added TextBlob to `requirements.txt` and added fallback guard

### 20.2 Compatibility [P2]

- [ ] **Python 3.8 CI** - Add to test matrix
- [ ] **Type hint compat** - `from __future__ import annotations`

### 20.3 Guards [P1]

- [ ] **Optional imports** - try/except for all optionals

---

## 21. Future Vision

> Source: `living_lexicon_discussion.md` - Long-term goals

### 21.1 Multi-Language Lexicons [P3]

- [ ] **Per-language tables** - Python, C++, Rust, JS lexicons
- [ ] **Cross-references** - Concept bridging across languages
- [ ] **Integration** - Unified with English lexicon

### 21.2 Tool Calling & Self-Modification [P3]

- [ ] **Tool API** - `create_file()`, `run_in_sandbox()`
- [ ] **Safe sandbox** - Container-based execution
- [ ] **Self-modification** - `edit_file()` with review

### 21.3 Dynamic Language Core [P3]

- [ ] **LoRA/PEFT adapters** - Incremental fine-tuning
- [ ] **Identity tokens** - Persona embeddings
- [ ] **Experience replay** - Continuous learning

### 21.4 Distributed [P3]

- [ ] **Celery/RQ** - Task queues
- [ ] **Milvus cluster** - Distributed vectors
- [ ] **Kubernetes** - Deployment configs

### 21.5 API & UI [P3]

- [ ] **REST API** - FastAPI
- [ ] **GraphQL** - Query interface
- [ ] **Web dashboard** - Analytics UI
- [ ] **Interactive explorer** - Graph visualization

---

## 22. Eidosian Checklist

> Source: `code_perfection_prompt.md`, `code_polishing_prompt.md`

### Verification Criteria

- [ ] All existing functionality preserved
- [ ] Performance metrics improved ≥30%
- [ ] Memory usage stable or reduced
- [ ] Public interfaces maintain compatibility
- [ ] Type coverage 100% for public APIs
- [ ] Error handling consistent across modules

### Quality Standards

- [ ] Further removal would break functionality
- [ ] Structure prevents invalid state by design
- [ ] Names reveal intent without comments
- [ ] Types enforce correctness at compile time
- [ ] Error handling explicit and informative
- [ ] Performance improvements measurable

---

## Priority Summary

| Priority | Count | Focus Area |
|----------|-------|------------|
| P0 | 4 | Tests, security, CI |
| P1 | 20 | Core features, docs, types |
| P2 | 45 | Architecture, performance |
| P3 | 22 | Future features, polish |

### Immediate Actions (P0)

1. ~~Create missing test files for workers~~ ✅
2. ~~Add shared test fixtures in conftest.py~~ ✅
3. ~~Mock heavy dependencies (torch, chromadb)~~ ✅
4. Audit SQL queries for parameterization
5. Add dependency version constraints

### Short-term (P1)

1. ~~Fix malformed docstrings~~ ✅
2. ~~Add `--version` CLI flag~~ ✅
3. Implement vector persistence verification
4. Add coverage to CI pipeline
5. ~~Consolidate exception definitions~~ ✅
6. ~~Add error codes to exceptions~~ ✅
7. ~~Add CLI vector search command~~ ✅
8. ~~Add CLI conversation commands~~ ✅
9. ~~Add README troubleshooting section~~ ✅
10. ~~Add README API quick reference~~ ✅

---

## Metrics Targets

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage | ~60% | 80%+ |
| Type coverage | ~80% | 95%+ |
| Doc coverage | ~70% | 100% public APIs |
| Max file lines | 1757 | <500 |
| Max complexity | ~15 | <10 |

---

*Document Version: 2.1*  
*Last Updated: 2026-01-11*  
*Status: Actively Maintained*
