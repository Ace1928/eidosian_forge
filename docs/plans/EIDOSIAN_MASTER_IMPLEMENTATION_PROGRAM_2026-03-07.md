# Eidosian Master Implementation Program

Date: 2026-03-07
Status: Living master program
Directive source: `docs/plans/AUTONOMOUS_EXECUTION_DIRECTIVE_2026-03-07.md`
Canonical backlog inventory:
- `reports/plans/plan_inventory_2026-03-07.json`
- `reports/plans/plan_inventory_2026-03-07.txt`
- `docs/plan_sweep/PLAN_SWEEP_TRACKER.md`
Active subordinate execution slices:
- `docs/plans/CODE_FORGE_GIS_EXECUTION_PROGRAM_2026-03-07.md`
- `docs/plans/LOCAL_AGENT_MCP_EXECUTION_PROGRAM_2026-03-07.md`
- `docs/plans/DIRECTORY_DOCUMENTATION_EXECUTION_PROGRAM_2026-03-13.md`
- `docs/plans/ENTITY_PROOF_EXECUTION_PROGRAM_2026-03-13.md`

## Program Intent

Unify all existing forge plans, TODOs, roadmaps, backlog trackers, and active infrastructure work into one production implementation program that:
- stabilizes CI and workflow execution,
- hardens Termux/Linux parity and service orchestration,
- upgrades Code Forge into the primary archive/code abstraction substrate,
- upgrades GIS into the universal identification and information-governance substrate,
- completes vector-native memory/knowledge/code/document/lexicon integration,
- drives GraphRAG through the entire information estate,
- and exposes the result through scheduler, autonomy, consciousness, and Atlas control planes.

## Baseline Facts

- Plan inventory currently reports `92` open markdown backlog files and `1096` open checklist items.
- Largest open backlog surfaces:
  - `word_forge`: 233 items
  - `game_forge`: 133 items
  - `computer_control_forge`: 74 items
  - `web_interface_forge`: 73 items
  - `eidos_mcp`: 72 items
- `archive_forge` currently contains approximately `722710` files and is too large to treat as a monolithic one-shot ingestion target.
- Current live CI failure classes observed from GitHub Actions:
  - `Eidosian Universal CI`: monolithic Python test job fails on forge-specific dependency drift and syntax regressions.
  - `Consciousness Linux Parity`: parity smoke fails in the smoke entrypoint and is too broad for reliable diagnostics.
- Current vector substrate is operational locally, but Linux CI still fails when `hnswlib` is absent and tests assume the backend is mandatory.
- Current GraphRAG native path exists, but lateral integration with Code Forge, doc pipelines, GIS, and lexicon pipelines is not yet complete enough to serve as the single global information substrate.

## Execution Rules

- Keep this file as the living top-level program.
- Check off completed items with markdown strike-through in subordinate artifacts and status sections.
- Every completed slice must update at least one of:
  - memory,
  - knowledge,
  - a canonical plan/status artifact,
  - CI/tests,
  - git history.
- Prefer narrow, composable commits with explicit artifact/test outcomes.
- Treat CI definitions as product code, not repo decoration.

## Program Phases

### Phase 0: Program Governance, Inventory, and Traceability
- [x] Freeze a canonical backlog snapshot from all `TODO`, `PLAN`, and `ROADMAP` documents.
- [ ] Generate a normalized master backlog table grouped by forge, domain, and execution dependency.
- [x] Create status links from this master program to all subordinate plans still in force.
- [ ] Mark obsolete or superseded plan artifacts instead of silently ignoring them.
- [ ] Create a unified completion ledger for commits, workflow runs, benchmarks, and ingestion milestones.

### Phase 1: CI and Workflow Stabilization
- [~] Replace monolithic CI test execution with forge-scoped matrices and reusable workflow calls.
- [~] Split Python testing into independently reportable components:
  - `agent_forge`
  - `memory_forge`
  - `knowledge_forge`
  - `code_forge`
  - `doc_forge`
  - `eidos_mcp`
  - `benchmarks`
  - `web_interface_forge`
  - additional forge groups as needed
- [~] Ensure every workflow reports:
  - exact failing forge/component,
  - exact failing command,
  - artifact links,
  - pass/fail summary table.
- [x] Make format automation targeted:
  - changed files only where possible,
  - scoped directories otherwise,
  - no global repo rewrites on unrelated pushes.
- [ ] Add a dependency-capability matrix so optional backends are explicit instead of becoming surprise CI failures.
- [~] Fix current known failure modes:
  - `eidctl.py` syntax drift regression gate,
  - missing `hnswlib` assumption in Linux CI,
  - oversized parity smoke blast radius,
  - secret-scan false positives/noisy paths if still present on latest runs.
- [~] Add artifact publication for per-forge test summaries, not just raw logs.
- [x] Add workflow self-tests and action pin governance without over-triggering on unrelated YAML.

### Phase 2: Termux/Linux Runtime Platform
- [~] Complete migration from shell-started services to supervised service management.
- [~] Keep one canonical shell bootstrap contract across Termux and Linux.
- [~] Finish boot/resume semantics and ensure resumable services restart cleanly.
- [~] Standardize temp/runtime/cache/data paths across Termux and Linux.
- [x] Define and audit runtime/generated artifact boundaries so generated state is distinguishable from source.
- [x] Build a capability registry for platform differences instead of littering checks across code.
- [x] Integrate boot/service/runtime state into Atlas and the scheduler.

### Phase 3: GIS Core and Universal Information Model
- [ ] Define GIS as the canonical identification, classification, and governance layer for all information constructs.
- [ ] Add a stable GIS identifier schema that can address:
  - code units,
  - files,
  - documents,
  - memory records,
  - lexicon terms,
  - graph communities,
  - benchmark artifacts,
  - source references,
  - pipeline runs.
- [ ] Define GIS namespaces, domain classes, relationship classes, and extension points.
- [ ] Add GIS-backed lookup/index APIs and deterministic serialization.
- [~] Link GIS identifiers into Knowledge Forge, Memory Forge, Code Forge, Word Forge, and GraphRAG artifacts.
- [ ] Add GIS-aware visualization and search in Atlas.

### Phase 4: Shared Vector and Embedding Substrate
- [~] Make the vector substrate a first-class shared service contract across memory, knowledge, code, documents, and lexicon.
- [ ] Standardize:
  - embedder interface,
  - vector-store interface,
  - metadata contract,
  - deletion/update semantics,
  - reindex semantics,
  - backend capability flags.
- [ ] Add backend selection policies for:
  - Termux-safe runtime,
  - Linux CI/runtime,
  - optional future backends.
- [~] Add vector drift detection, dimension validation, and rebuild policy.
- [ ] Add benchmark coverage for recall, latency, persistence, rebuild time, and metadata filtering.

### Phase 5: Memory Forge Completion
- [~] Finish migration of all memory tiers to the shared vector substrate.
- [ ] Ensure every memory record has:
  - GIS identity,
  - embedding/vector metadata,
  - tags/domains/keywords,
  - community assignment,
  - provenance,
  - update timestamps,
  - dedupe signature.
- [ ] Add LLM-assisted enrichment policies for selective high-value memories.
- [~] Add community drift tracking, tier-health tracking, and quality scoring.
- [~] Feed memory trends into autonomy, consciousness, Atlas, and scheduler policy.
- [x] Normalize timezone handling across tier persistence and introspection so live/report comparisons remain stable across runtimes.

### Phase 6: Knowledge Forge and Native GraphRAG Completion
- [~] Complete GraphRAG-native indexing parity for code, docs, memory, lexicon, references, and artifacts.
- [~] Ensure native report, trend, and assessment layers cover all indexed domains.
- [~] Add GIS and vector identity to graph nodes and communities.
- [ ] Add richer graph operations:
  - multi-hop traversal,
  - community quality scoring,
  - contradiction/gap analysis,
  - evidence lineage.
- [~] Integrate crawl/Tika/doc ingestion outputs as first-class knowledge artifacts.

### Phase 7: Code Forge Deep Upgrade
- [~] Upgrade Code Forge into the primary abstraction layer for archive reduction and code reuse.
- [ ] Expand reversibility and canonicalization for extracted/archive code.
- [ ] Improve deduplication across:
  - exact copies,
  - normalized copies,
  - structural clones,
  - semantic near-duplicates,
  - migrated abstractions.
- [ ] Add canonical extraction and abstraction workflows for reusable library synthesis.
- [ ] Add standardization passes for naming, signatures, ownership, provenance, and test linkage.
- [~] Improve vectorized search, reverse lookup, snippet extraction, and dependency graph navigation.
- [ ] Ensure archive-derived code can be ingested into the library and then safely retired from raw archive storage when replacement gates pass.

### Phase 8: Archive Forge Reduction Program
- [~] Classify `archive_forge` into ingestion batches by source, type, and expected retention strategy.
- [~] Ingest code-like material into Code Forge first.
- [~] Ingest document-like material into doc/knowledge/GraphRAG pipelines.
- [~] Ingest reference/metadata/manifests into GIS/knowledge provenance stores.
- [ ] Add promotion/deletion gates so raw archive content is only removed after abstraction and evidence thresholds are met.
- [~] Track archive burn-down with artifacts and dashboards.

### Phase 9: Document, Tika, and Crawl Completion
- [~] Harden Tika and crawl ingestion for all supported local and fetched artifacts.
- [~] Create production pipelines for leftover non-code documents and file types.
- [~] Ensure final docs from doc processing are re-ingested into knowledge/GraphRAG.
- [~] Add source-reference ingestion as a first-class workflow with provenance, GIS IDs, and vectors.
- [~] Add file-type-specific routing policies and extraction QA metrics.
- [~] Add managed directory documentation, coverage APIs, batch README generation, and runtime health integration across the forge estate.
- [~] Add Atlas-side documentation navigation, batch regeneration, and drift history as operator-facing controls.
- [~] Add documentation governance contracts for suppressed directories, review gates, and promotion readiness.
- [x] Add identity continuity history/trend views to Atlas, not just latest-score surfaces.

### Phase 10: Word Forge and Living Lexicon Completion
- [~] Ensure every new term/phrase discovered by code/doc/crawl/memory pipelines enters the lexicon queue.
- [~] Expand term enrichment to include:
  - definitions,
  - examples,
  - part of speech,
  - roots,
  - phonetics,
  - domain tags,
  - GIS identity,
  - graph relationships,
  - community membership.
- [ ] Connect lexicon communities to knowledge and code communities.
- [~] Expose the queue, processing state, and graph in Atlas.

### Phase 11: Scheduler, Coordinator, Autonomy, Consciousness
- [~] Make every model-using component obey the coordinator budget contract.
- [x] Ensure interactive sessions preempt non-interactive work safely.
- [~] Make scheduler jobs persistent, resumable, interrupt-safe, and stateful across restarts.
- [~] Feed runtime, ingestion, memory, report, and vector health into autonomy scoring.
- [~] Feed the same state into consciousness broadcasts and metrics.
- [~] Add explicit backlog and resume semantics for unfinished ingestion waves.
- [x] Add governed local-agent execution as a first-class runtime surface.
- [x] Add explicit unit coverage for new consciousness runtime modules (`default_mode`, `metacognition`, `motor`) and reconcile newly added routers against focused tests.
- [x] Add cross-interface continuity import/synchronization across Codex, Gemini, qwenchat, and local-agent session surfaces.
- [x] Make narrative anchoring and desire-state introspection explicit, configurable, and test-covered instead of hidden inline behavior.
- [~] Normalize MCP boundary coercion so list-like router parameters tolerate stable cross-client inputs instead of brittle schema mismatches.

### Phase 12: Atlas and Operator UX
- [~] Expand Atlas into the operator control plane for:
  - services,
  - pipelines,
  - graph exploration,
  - lexicon,
  - code library,
  - memory communities,
  - workflow history,
  - GIS lookup.
- [~] Add saved exploration sessions, filters, community overlays, and reversible snippet export.
- [~] Add runtime trend/history panes for workflows, pipelines, vector state, and archive burn-down.
- [~] Add local-agent runtime status/history to the operator surface.
- [~] Add execution control for supervised services and scheduler queues.
- [~] Add documentation coverage/diff/regeneration controls to Atlas.

### Phase 13: Validation, Benchmarks, and Promotion Gates
- [ ] Define end-to-end promotion gates for each subsystem.
- [ ] Add benchmark suites for:
  - vector recall/latency,
  - graph/community quality,
  - Code Forge reuse/retrieval quality,
  - doc quality,
  - lexicon enrichment quality,
  - external agent suites with publishable artifacts.
- [~] Add externally reportable benchmark evidence:
  - [x] imported AgencyBench reference artifact
  - [x] imported AgentBench leaderboard reference artifact
  - [x] live AgencyBench scenario2 deterministic artifact
  - [x] live AgencyBench scenario1 deterministic artifact
  - [x] live runtime benchmark observability artifacts and Atlas/API surfaces
  - [ ] live non-deterministic model-driven external benchmark artifact
  - scheduler throughput,
  - service resiliency.
- [ ] Require green CI, green targeted regression suites, and artifact publication before promotion/deletion actions.
- [~] Add an externally legible proof scorecard that aggregates capability, continuity, governance, observability, reproducibility, and adversarial evidence into one reportable contract.
- [~] Add freshness/regression proof policy and migration/replay contracts:
  - [x] freshness degradation for stale evidence
  - [x] previous-scorecard regression deltas
  - [x] migration/replay scorecard contract
  - [x] live imported mainstream external benchmark evidence
  - [x] second mainstream imported benchmark suite evidence
  - [x] identity continuity trend/history included in proof artifacts
  - [x] proof history trend/delta included in proof artifacts
  - [x] session-bridge continuity evidence included in proof artifacts and bundle

## Immediate Execution Order

1. [~] Stabilize CI/workflows and make failures forge-specific and actionable.
2. [~] Complete the runtime/platform/service substrate so long-running ingestion can stay up.
3. [~] Deepen Code Forge and vector contracts enough to process archive code safely.
4. [~] Build GIS identifier/model integration across memory/knowledge/code/docs/lexicon.
5. [~] Run archive/document ingestion waves with GraphRAG/native graph/vector integration.
6. [~] Complete Atlas/operator control and autonomous resumption loops.

## Progress Log

- [x] Backlog inventory generated:
  - `reports/plans/plan_inventory_2026-03-07.json`
  - `reports/plans/plan_inventory_2026-03-07.txt`
- [x] Directive preserved verbatim:
  - `docs/plans/AUTONOMOUS_EXECUTION_DIRECTIVE_2026-03-07.md`
- [x] Research/source program created:
  - `docs/plans/EIDOSIAN_RESEARCH_AND_SOURCE_PROGRAM_2026-03-07.md`
- [x] Documentation governance slice extended with:
  - suppression/override policy
  - high-risk README review gates
  - scheduler/coordinator/autonomy/daemon integration
  - Atlas review/suppression visibility
- [x] Entity proof slice started with:
  - canonical proof scorecard generator
  - Atlas/API exposure of latest proof report
  - explicit missing-evidence reporting for external validity gaps
- [x] Entity proof slice extended with:
  - freshness and regression scoring
  - external benchmark import contract
  - AgencyBench official sample-reference import path and artifact
  - migration/replay scorecard contract
  - scheduler runtime proof normalization
  - identity continuity history/trend publication in scorecards and bundles
- [x] Atlas/proof slice extended with:
  - `/api/proof/summary`
  - `/api/proof/history`
  - `/api/proof/external`
  - Atlas proof-history and external-benchmark tables
  - bundle inclusion of session-bridge runtime evidence
- [x] Runtime benchmark observability slice extended with:
  - `status.json`, `attempts.jsonl`, and `model_trace.jsonl` per live AgencyBench run
  - `/api/benchmarks/runtime`
  - Atlas runtime benchmark table
  - proof scorecard and bundle inclusion of runtime benchmark status evidence
- [x] Initial primary-source set saved and ingested:
  - `docs/external_references/2026-03-07-master-program/`
- [x] Phase 1 implementation slice started:
  - dynamic Python component test-matrix planner added
- [x] Directory documentation execution slice started and broad README coverage generated:
  - `docs/plans/DIRECTORY_DOCUMENTATION_EXECUTION_PROGRAM_2026-03-13.md`
  - `reports/docs/directory_docs_postwave_all.json`
- [x] Directory documentation now feeds Atlas/runtime/autonomy integration:
  - `data/runtime/directory_docs_status.json`
  - Atlas `/api/docs/*`
  - supervisor/runtime coverage signals
- [x] Phase 2 runtime-control slice deepened:
  - runtime capability registry is now emitted as a canonical artifact
  - Termux boot install ensures runit service definitions exist before first boot use
  - Atlas now exposes supervised service state and scheduler control surfaces
  - scheduler pause/resume/stop controls now exist as explicit runtime operations
- [x] Consciousness bridge continuity hardened:
  - status tool now falls back to direct bridge probing when recent events are absent
  - bridge and self-model defaults now point to `data/tiered_memory`
  - live MCP reload confirmed real bridge visibility (`memory_total=219`, `knowledge_count=4595`)
- [x] Termux shell bootstrap thinned and modularized:
  - repo-managed `~/.bashrc` bootstrap
  - restored `eidosian_venv/bin/activate`
  - startup backup/audit artifacts
- [~] Runit-backed Termux service migration advanced:
  - `scripts/install_termux_runit_services.sh`
  - `scripts/eidos_termux_services.sh` now prefers `sv` when installed
  - scheduler state persistence and recovery added
  - runit service definitions installed under `$PREFIX/var/service` with `down` guards for safe boot handoff
- [~] Termux launch contract hardened:
  - forge `scripts/` added to the canonical shell/env path contract
  - interactive startup now exports Atlas URL and requests wake-lock
  - Atlas now exposes forge/home browsing and service control APIs
  - Universal CI Python tests refactored toward per-component matrix execution
  - Linux parity smoke split into explicit phases (`pytest`, `stdio`, `audit`)
- [x] Local-agent control-plane slice advanced:
  - guarded local MCP agent now records transport/resource telemetry
  - local-agent state is visible in Atlas and autonomy
  - scheduler wrapper/service path implemented and bounded live smoke now reaches the real living pipeline
  - official MCP control-plane + Ollama residency/context references saved and ingested
- [x] CI workflow scope control expanded:
  - reusable changed-manifest planner added for Python, Prettier, and component lint scopes
  - `format.yml` converted from monorepo-wide formatting to changed-scope formatting
- [x] Remaining Universal CI breakages patched in local and remote workflow code:
  - `eidctl.py` syntax regression repaired
  - `Markdown` added to CI dependency bootstrap
  - `lib/eidosian_runtime/__init__.py` added so runtime imports resolve in CI
- [x] Code Forge / GIS implementation slice created:
  - `docs/plans/CODE_FORGE_GIS_EXECUTION_PROGRAM_2026-03-07.md`
- [x] Local agent / MCP implementation slice created:
  - `docs/plans/LOCAL_AGENT_MCP_EXECUTION_PROGRAM_2026-03-07.md`
- [x] Code Forge / GIS primary-source set saved and ingested:
  - `docs/external_references/2026-03-07-code-forge-gis/`
  - `lint.yml` converted from monorepo-wide linting to per-component changed-scope linting
  - `secret-scan.yml` now uploads a SARIF artifact and ignores mirrored external reference snapshots
  - Universal CI format/lint passes now consume the same changed-manifest contract
- [x] Workflow maintenance surfaces tightened:
  - `workflow-lint.yml` now detects changed workflow files before running `actionlint`
  - `security-audit.yml` now publishes generated reports as artifacts
  - CI source set extended with actionlint and gitleaks primary references and ingested locally
- [x] Live CI regression follow-up applied after first scoped rollout:
  - tracked the missing `scripts/ci/python_test_matrix.py` and regression test
  - changed-manifest component selection no longer expands all Python components for workflow-only changes
  - changed-file lint now operates on changed files within each component instead of whole component roots
  - auto-format no longer fails on residual non-fixable Ruff findings after auto-fix
- [x] Archive-code/document/metadata ingestion moved from planning into bounded live execution:
  - resumable archive-wave execution implemented
  - bounded include-path execution avoids full-tree rescans
  - live real-archive smoke succeeded with `3` selected batches and `0` failures
- [x] Code Forge export quality tightened for GraphRAG / Knowledge Forge promotion:
  - structural-noise units are filtered by default
  - live bounded rerun cut elapsed time from `25.405s` to `13.209s` (`48.01%` reduction)
- [x] Guarded local small-agent substrate started:
  - source bundle saved under `docs/external_references/2026-03-07-local-agent-mcp/`
  - source bundle ingested locally via Tika-backed ingestion (`12 files`, `102 nodes`)
  - policy-contracted MCP/Qwen harness added under `lib/eidosian_agent/`
  - CLI wrapper added at `scripts/eidos_local_agent.py`
- [x] Local small-agent service and validation hardening advanced:
  - continuous service wrapper added at `scripts/run_local_mcp_agent.sh`
  - `eidos_termux_services.sh` can now manage the local agent
  - stale-own-lease recovery prevents stuck coordinator ownership after failed runs
  - live bounded observer validation now returns structured timeout artifacts instead of crashing on long Qwen turns

## Current Known Blocking Defects

- [x] Universal CI Python testing is now broken out into per-component matrix jobs.
- [ ] Linux CI assumes or encounters `hnswlib` unavailability in code paths that should be capability-gated.
- [ ] Some workflow automation is still broader than ideal outside the newly scoped format/lint/test surfaces.
- [ ] Archive ingestion scale is not yet chunked and governed enough for safe burn-down.
- [ ] GIS is still conceptually present but not yet the enforced canonical identity/governance layer.

## Completion Discipline

For each completed item:
- update the relevant subordinate plan/TODO/roadmap,
- update this master program,
- add/refresh tests and artifacts,
- ingest any new source/reference material,
- store a lesson or memory if the step changed the operating model,
- commit and push a narrow change set.
