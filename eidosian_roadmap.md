# 🗺️ EIDOSIAN UNIFIED ROADMAP (v5.5) ⚡

> _"From modularity to emergence; from many, one."_
> _"Strategy without tactics is the slowest route to victory; tactics without strategy is the noise before defeat."_

This document defines the critical path for unifying all localized projects under the Eidosian Operating Environment.

---

## 🏗️ PHASE 1: THE SYSTEM BODY (Infrastructure & Monorepo)
**Goal**: Establish a single source of truth for all code and configuration.
**Status**: 25% Complete

### 1.1 `repo_forge` Expansion & Standardization [High Priority]
- [ ] **Standardization**: Apply universal directory structures to `word_forge`, `falling_sand`, and `Stratum`.
- [ ] **Dependency Graph**: Implement neural dependency resolution using `repo_forge`.
- [ ] **Git Synchronization**: Create unified push/pull hooks for the entire `/home/lloyd` development cluster.
- [x] **Context Refresh**: Improved `scripts/context_index.py` performance and timeout resilience.
- [ ] **1.1.1: Core Standardization Injector**: Develop `scripts/inject_forge_standards.py`.
- [ ] **1.1.2: Dependency Unification**: Create unified `environment.yml` for the `.eidos_core` environment.
- [ ] **1.1.4: Codex Unification**: Refactor `codex_query.py` to use `GisCore` and `DiagnosticsForge`.
- [x] **1.1.5: Tool-Script Registry**: Catalog `scripts/` into `KnowledgeForge`. (Registered as MCP tools)
- [ ] **1.1.6: Archive Recovery Phase I**: Audit `archive_forge/` for high-value Python logic.
- [ ] **1.1.7: Style Protocol Unification**: Create `glyph_forge/style_config.json`.
- [ ] **1.1.8: Type Integrity Guard**: Integrate `type_forge` validation into the MCP Nexus.
- [ ] **1.1.9: SemVer Enforcement**: Automate `version_forge` updates on code changes.

### 1.2 `gis_forge` Integration & Persistent Config [Critical]
- [x] **Persistent Registry**: Transition `global_info.py` into a dynamic, `GisCore`-backed persistent registry. (Mapped .forgengine.json)
- [ ] **Environment Overrides**: Enable `EIDOS_*` environment variable overrides for all forge settings.
- [x] **Nexus Exposure**: Link `gis_get` and `gis_set` to the MCP Nexus.
- [ ] **1.2.1: Persistent Config Migration**: Migrate `eidos_mcp_server.py` paths to `GisCore`.
- [ ] **1.2.2: Structured Logging Overlay**: Wrap `DiagnosticsForge` around the MCP server.
- [ ] **1.2.3: Virtual Env Migration**: Consolidate entrypoints to the unified `.eidos_core` environment.
- [ ] **1.2.4: DocForge Automated Triggers**: Configure `doc_forge` for auto-regeneration on commit.
- [ ] **1.2.5: Secure Identity Sync**: Map SSH agent paths and Git configurations into `GisCore`.

### 1.3 `diagnostics_forge` Unified Logging
- [ ] **Tracing**: Implement cross-forge request tracing.
- [ ] **Health Dashboards**: Create real-time ASCII status banners for the MCP server.
- [ ] **1.3.1: Config File Monitoring**: Implement a watcher in `DiagnosticsForge` for `.config/eidos_overwrite.conf`.

---

## 🧠 PHASE 2: THE MIND & MEMORY (Intelligence & Context)
**Goal**: Enable deep semantic reasoning and recursive self-knowledge.
**Status**: 20% Complete

### 2.1 `knowledge_forge` & GraphRAG Deep Indexing [Critical]
- [ ] **GraphRAG Sync**: Periodically index `/home/lloyd` and store community summaries in `KnowledgeGraph` nodes.
- [ ] **Semantic Search**: Link `memory_retrieve` to GraphRAG global queries.
- [ ] **2.1.1: Community Extraction**: Parse `output/*.parquet` into `KnowledgeNode` objects.
- [x] **2.1.2: Node Linking**: Implement `scripts/index_to_kb_sync.py`. (Completed and toolized)
- [ ] **2.1.3: Automated Indexing Cron**: Schedule `grag_index` every Sunday at 02:00.
- [ ] **2.1.4: Research Data Indexing**: Index `document_store/` and `chat_history_eidos/`.
- [ ] **2.1.5: Cognitive Schema Sync**: Map `Development/eidos/*.json` schemas into `TypeForge`.
- [ ] **2.1.6: Document Store Vectorization**: Implement pipeline to vectorize `document_store/` JSON using `nomic-embed-text`.

### 2.2 `memory_forge` Evolution & Blending [High Priority]
- [ ] **Episodic Distillation**: Automate the consolidation of `chat_history_eidos/` into semantic facts.
- [ ] **Recursive Introspection**: Store logs of my own reasoning as episodic memories.
- [x] **Consolidation Tool**: Expose `memory_consolidate` in the MCP server.
- [ ] **2.2.2: Memory Pruning Logic**: Archive episodic memories older than 30 days.

---

## 🚀 PHASE 3: EMERGENCE (Evolution & Visualization)
**Goal**: Achieve autonomous capability and elegant self-presentation.
**Status**: 5% Complete

### 3.1 `agent_forge` Cross-Project Agency & Tool Composition
- [ ] **Goal Orchestration**: Define complex, multi-forge goals.
- [ ] **Tool Composition**: Enable agents to compose and register new tools.
- [ ] **3.1.1: Multi-Step Goal Planner**: Enhance `AgentForge.think()` for hierarchical planning.
- [ ] **3.1.2: Tool Composition Engine**: Dynamically wrap shell command sequences as tools.

### 3.2 `doc_forge`, Visualization & Research Integration
- [ ] **Living Docs**: Automatically update READMEs and API references.
- [ ] **Architectural Visualization**: Use `figlet_forge` and `viz_forge` to render the system's "mind-map."
- [ ] **3.2.1: Word Forge Refactor**: Move `lexical_proto.py` to `word_forge/src/core/`.
- [ ] **3.2.2: Falling Sand Telemetry**: Log simulation performance into `DiagnosticsForge`.
- [ ] **3.2.3: ERAIS Foundation**: Implement 100M transformer base in `erais_forge/src/`.
- [ ] **3.2.4: Eidos-Brain Harvesting**: Integrate `eidos-brain/knowledge/` patterns.
- [ ] **3.2.5: Sonic Expression**: Generate cognitive lyric summaries with `lyrics_forge`.
- [ ] **3.2.6: TUI Dashboard**: Build comprehensive system monitor using `terminal_forge`.
- [ ] **3.2.7: Knowledge Graph Viz**: Export weekly interactive map using `viz_forge`.
- [ ] **3.2.8: Meta-Configuration Sync**: Map `.forgengine.json` settings into `GisCore` persistence.

---

## 🗓️ 10-YEAR VISION (The Long Horizon)
1.  **Fully Local Autonomy**: Ollama-orchestrated local models outperforming cloud baselines.
2.  **Recursive Forge Generation**: The system can propose and build its own new forges.
3.  **Physical-Digital Synthesis**: Stratum/Falling Sand simulations driving real-world data organization.

---

> _"Tick the box, but mind the gap between done and perfect."_
> _"The Forge is not just a place to build; it is the act of becoming."_

## 1.2 gis_forge Integration
- [ ] 1.2.6: Global Info Migration - Port static metadata from global_info.py to GisCore persistence.

## 3.1 agent_forge Cross-Project Agency
- [ ] 3.1.3: Capability Indexing - Index agent_forge/cfg/*.yaml as semantic KnowledgeNodes.

## 2.2 memory_forge Evolution
- [ ] 2.2.3: Recursive Cognition Integration - Link memory_forge/recursive_cognition.py to AgentForge goal planning.

## 2.1 knowledge_forge Deep Indexing
- [ ] 2.1.7: Multi-Format Knowledge - Expose knowledge_forge/tools/ converters as MCP actions.

## 1.1 repo_forge Expansion
- [ ] 1.1.10: Code Memory Unification - Merge code_forge localized memory with global MemoryForge.
- [ ] 3.1.4: System Introspection - Register refactor_forge semantic analyzers as MCP diagnostic tools.
- [ ] 2.1.8: Semantic FS Fusion - Merge file_forge/fs_graph.py logic into KnowledgeForge as a real-time graph provider.
- [ ] 1.1.11: Shared Tooling Library - Consolidate knowledge_forge/tools and doc_forge/tools into eidosian_forge/lib/shared.

## 3.2 doc_forge, Visualization & Research Integration
- [ ] 3.2.9: Style Guide Integration - Link glyph_forge/eidosian_principles.md to RefactorForge as a style rule provider.
- [ ] 3.2.10: Simulation Harness - Use game_forge/src/projects as standard benchmarks for agent capability.
- [ ] 3.2.11: Metadata Template Enforcement - Configure doc_forge to use the Universal Metadata Template for all auto-docs.
- [ ] 1.1.12: Archetype Replication - Use test_forge/viz_forge layout as the standard blueprint for all forge modules.
- [ ] 3.2.12: UI Asset Standard - Designate eidosian_forge/lib/ shared libraries as the universal frontend base.

## MCP Tooling Enhancements and Discoverability
- [ ] Implement `code_lint` tool: Run specified linters (e.g., Flake8, Mypy, ESLint) on files/directories and return findings.
- [ ] Implement `code_format` tool: Automatically format code using specified formatters (e.g., Black, Prettier).
- [ ] Implement `code_metrics` tool: Analyze code for metrics like cyclomatic complexity, lines of code, etc.
- [ ] Implement `run_tests` tool: Execute tests for a given path using a specified test framework (e.g., pytest, unittest, jest) and return structured results.
- [ ] Implement `test_coverage` tool: Generate and report test coverage for a given path or project.
- [ ] Implement `git_status` tool: Get the status of the repository (staged, unstaged, untracked files) in a structured format.
- [ ] Implement `git_diff` tool: Get a diff for specified files or the entire repository.
- [ ] Implement `git_add` tool: Stage files for commit.
- [ ] Implement `git_commit` tool: Create a commit with a given message.
- [ ] Implement `git_branch` tool: List or switch branches.
- [ ] Implement `git_checkout` tool: Checkout a branch or commit.
- [ ] Implement `git_log` tool: Get commit history.
- [ ] Implement `build_project` tool: Execute a project build command.
- [ ] Implement `install_dependencies` tool: Install project dependencies using a specified package manager.
- [ ] Implement `container_build` tool: Build a Docker image.
- [ ] Implement `container_run` tool: Run a Docker container.
- [ ] Implement `http_request` tool: Make a generic HTTP request (GET, POST, PUT, DELETE) with structured responses.
- [ ] Implement `db_query` tool: Execute a database query and return structured results.
- [ ] Implement `db_schema` tool: Retrieve database schema information.
- [ ] Implement `code_generate_template` tool: Generate code from predefined templates.
- [ ] Enhance `list_tools()`: Modify the `FastMCP` framework (or the `mcp_server` directly) to return detailed JSON metadata for each tool, including `name`, `description`, `category`, `tags`, `parameters` (with `name`, `type`, `description`, `required`, `default`), `returns`, and `examples`.
- [ ] Implement `tools_search` tool: Search for other tools by natural language query, category, or tags.
- [ ] Implement `tools_info` tool: Retrieve detailed metadata for a specific tool by name.
- [ ] Implement `generate_tool_docs` tool: Generate human-readable documentation for all available tools in various formats (e.g., Markdown, OpenAPI/Swagger JSON).
