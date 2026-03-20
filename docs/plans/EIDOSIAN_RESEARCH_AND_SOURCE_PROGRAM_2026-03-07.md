# Eidosian Research and Source Program

Date: 2026-03-07
Purpose: Define the source collection, validation, and ingestion program that supports the master implementation program.

## Research Rules

- Prefer primary sources:
  - official documentation,
  - official repositories,
  - standards/specifications,
  - maintainer-authored design docs.
- Save local copies of every high-value reference under dated `docs/external_references/...` directories.
- Ingest saved references into the knowledge/vector pipeline via Tika.
- Track source-to-phase linkage explicitly.
- Separate stable architectural sources from volatile release/status sources.

## Source Domains By Program Phase

### R0: CI and Workflow Engineering
Need sources for:
- GitHub Actions workflow syntax
- reusable workflows
- matrix strategies
- path filters
- concurrency/cancel behavior
- job summaries/artifacts
- action pinning and workflow linting
- pytest reporting/selection patterns

Primary source targets:
- `docs.github.com/actions/...`
- `github.com/rhysd/actionlint`
- `docs.pytest.org/...`

### R1: Termux/Linux Runtime and Services
Need sources for:
- Termux execution environment
- termux-services/runit
- termux-boot
- RUN_COMMAND / external app control
- termux-x11
- shell portability practices
- temp/runtime path semantics

Primary source targets:
- Termux official repos/wiki pages
- POSIX/Open Group environment variable docs
- Linux dynamic linker docs where preload behavior matters

### R2: GIS / Information Classification Layer
Need sources for:
- formal identifier design patterns
- classification / taxonomy / ontology practices
- graph identity and metadata schemas
- provenance and lineage modeling

Primary source targets:
- W3C RDF/OWL/PROV material
- schema/identifier best-practice docs
- relevant SQLite/JSON schema references

### R3: Shared Vector / ANN / Metadata Store
Need sources for:
- HNSW algorithm implementation and operational parameters
- persistence/update/delete semantics
- SQLite integration patterns
- metadata filtering approaches
- embedding/vector evaluation methodology

Primary source targets:
- `nmslib/hnswlib`
- SQLite official docs (WAL, FTS5, JSON1)
- evaluation/ANN primary documentation where needed

### R4: Memory and Knowledge Systems
Need sources for:
- graph storage patterns
- semantic retrieval patterns
- vector + graph hybrid retrieval
- memory/community clustering and evaluation

Primary source targets:
- official GraphRAG repo/docs
- NetworkX / graph library docs if directly used
- internal prior docs already in repo

### R5: Code Forge / Archive Reduction
Need sources for:
- tree-based code parsing and structural analysis
- clone detection / code similarity techniques
- canonicalization and refactoring provenance patterns
- repository-scale code indexing techniques

Primary source targets:
- Tree-sitter docs
- language parser/tooling docs in actual use
- official docs for any vector/code-index dependencies adopted

### R6: Document, Tika, and Crawl Pipelines
Need sources for:
- Apache Tika configuration and parser behavior
- extraction quality controls
- crawler integration and content normalization
- file-type routing and metadata handling

Primary source targets:
- Apache Tika official docs
- docs for any parser libraries directly used

### R7: Word Forge / Lexicon
Need sources for:
- lexical/semantic graph modeling
- dictionary/phonetic/morphological source formats
- enrichment pipelines and metadata design

Primary source targets:
- official lexical resource docs used by the codebase
- language-resource format specs where applicable

### R8: Dashboard / Atlas / UX / Visualization
Need sources for:
- force-graph and interactive graph UX patterns
- service-control UI patterns
- search/filter interaction models

Primary source targets:
- official docs for the frontend/chart/graph libraries actually in use

### R9: Local Small Agent / MCP Tool Contracts
Need sources for:
- MCP tool and resource semantics
- MCP client transport and authorization expectations
- local-model tool calling and structured-output patterns
- contract-driven tool schema restriction and guarded execution loops

Primary source targets:
- `modelcontextprotocol.io`
- `docs.ollama.com`
- official model pages under `ollama.com/library/...`

## Source Acquisition Waves

### Wave A: Immediate implementation blockers
- [x] GitHub Actions docs and workflow debugging references
- [x] GraphRAG official indexing/query docs
- [x] hnswlib official docs/README
- [x] SQLite WAL/FTS5/JSON1 docs
- [x] Apache Tika official docs
- [x] Tree-sitter official docs
- [x] pytest execution/reporting references
- [x] Secret-scanning workflow references and gitleaks configuration docs already captured through the GitHub Actions source set

### Wave B: Architectural unification
- [x] W3C provenance / RDF / OWL references for GIS and graph identity
- [ ] additional vector/metadata filtering references if backend changes are needed
- [ ] code-clone / parsing references if current internal approach is insufficient
- [~] external-proof benchmark/governance references for:
  - external agent benchmarks
  - observability / telemetry
  - adversarial failure analysis
  - publishable proof framing
  - migration/replay reproducibility doctrine
  - benchmark freshness/regression policy
  - GitHub workflow execution and benchmark-operation references for live AgencyBench scenario execution
  - AgentBench leaderboard/reference artifacts

### Wave C: UX / operator plane / runtime control
- [~] official docs for current dashboard stack
- [~] any service-control/websocket/runtime telemetry references directly tied to implementation
- [x] documentation-system references captured for managed README generation and Doc Forge documentation APIs
- [x] documentation governance / review references captured for suppression, review-gate, and ownership follow-up design

### Wave D: Local agent / MCP tool use
- [x] MCP tool/resource/client/authorization references
- [x] Ollama tool-calling / structured-output / thinking references
- [x] OpenClaw integration reference captured as a comparative local-agent pattern
- [~] Local-agent evaluation and safety references beyond the current implementation stack if a stronger harness is needed
- [x] Additional MCP control-plane references:
  - prompts
  - roots
  - sampling
  - Ollama FAQ keep-alive/runtime residency
  - Ollama context-length guidance

## Ingestion Protocol

For every saved source set:
- create a dated directory under `docs/external_references/`
- include a `README.md` describing source purpose and phase mapping
- ingest files through the Tika pipeline
- attach tags describing domain, phase, and subsystem

## Research Deliverables

- [ ] source manifest by phase
- [x] saved local source directories
- [x] ingested source artifacts in knowledge/vector system
- [x] implementation notes back-linked to source sets for the CI/workflow stabilization slice
- [x] updated master implementation program with sourced constraints and choices for the current CI slice
- [x] local-agent/MCP source program linked into an implementation slice

## Current Source Sets

- `docs/external_references/2026-03-07-termux-upgrade/`
- `docs/external_references/2026-03-07-tmp-redirection/`
- `docs/external_references/2026-03-07-master-program/`
- `docs/external_references/2026-03-07-master-program/ci/`
- `docs/external_references/2026-03-07-code-forge-gis/`
- `docs/external_references/2026-03-07-notebook-ingestion/`
- `docs/external_references/2026-03-07-archive-doc-routing/`
- `docs/external_references/2026-03-07-local-agent-mcp/`
- `docs/external_references/2026-03-07-local-agent-control-plane/`
- `docs/external_references/2026-03-07-termux-runit-services/`
- `docs/external_references/2026-03-07-runtime-path-contracts/`
- `docs/external_references/2026-03-13-directory-doc-system/`
- `docs/external_references/2026-03-13-doc-governance-review/`
- `docs/external_references/2026-03-13-entity-proof/`
- `docs/external_references/2026-03-13-proof-governance/`
- `docs/external_references/2026-03-13-agencybench/`
- `docs/external_references/2026-03-13-github-cli-agencybench-live/`
- `docs/external_references/2026-03-20-agentbench/`
