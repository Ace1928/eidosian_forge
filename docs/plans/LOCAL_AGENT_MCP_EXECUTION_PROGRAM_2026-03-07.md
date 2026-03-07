# Local Agent / MCP Execution Program

Date: 2026-03-07
Status: Living implementation slice
Parent program: `docs/plans/EIDOSIAN_MASTER_IMPLEMENTATION_PROGRAM_2026-03-07.md`

## Intent

Upgrade the local small-agent path into a production-grade, coordinator-aware, contract-driven harness that:
- uses the local Qwen runtime,
- discovers Eidos MCP tools dynamically,
- exposes only policy-contracted tool schemas to the model,
- validates every tool call before execution,
- records auditable cycle artifacts,
- and can run continuously without bypassing the coordinator.

## Primary Sources

Saved under `docs/external_references/2026-03-07-local-agent-mcp/` and intended for ingestion into knowledge.

- MCP tools: `mcp-tools.html`
- MCP resources: `mcp-resources.html`
- MCP client concepts: `mcp-client-concepts.html`
- MCP authorization: `mcp-authorization.html`
- Ollama tool calling: `ollama-tool-calling.html`
- Ollama structured outputs: `ollama-structured-outputs.html`
- Ollama thinking: `ollama-thinking.html`
- Ollama chat API: `ollama-api-chat.html`
- Ollama OpenAI compatibility: `ollama-openai-compatibility.html`
- Ollama OpenClaw integration: `ollama-openclaw.html`
- Ollama Qwen 3.5 2B: `ollama-qwen35-2b.html`

## Tracks

### Track A: Contract and Policy Surface
- [x] Define a versioned local-agent profile contract.
- [x] Add explicit per-tool allowlists and argument constraints.
- [x] Distinguish read-only vs mutating tool budgets.
- [ ] Add per-profile path sandboxes for any future file-bearing tools.
- [ ] Add signed or hashed policy snapshots for tamper-evident runtime verification.

### Track B: MCP Integration
- [x] Discover MCP tools dynamically at runtime.
- [x] Contract discovered schemas down to policy-allowed arguments only.
- [x] Support MCP HTTP transport first and stdio fallback second.
- [ ] Add richer resource usage alongside tool usage.
- [ ] Add transport telemetry and retry classification to cycle artifacts.

### Track C: Local Model Harness
- [x] Use the dedicated local Qwen runtime.
- [x] Support reasoning-on first with reasoning-off fallback for empty turns.
- [x] Implement a bounded agent loop with explicit max-step and max-tool-call ceilings.
- [ ] Add prompt/response benchmark artifacts for observer vs curator profiles.
- [ ] Add live reliability smoke over the dedicated qwen service and running MCP server.

### Track D: Runtime Governance
- [x] Enforce coordinator leasing around local-agent cycles.
- [x] Persist current status and append cycle history artifacts.
- [~] Register local-agent work as a first-class scheduled surface.
- [ ] Feed local-agent artifact health back into autonomy and Atlas.

## Immediate Queue

1. [x] Create guarded local-agent package, CLI, and policy profiles.
2. [~] Run live bounded observer-cycle validation against the active MCP service.
3. [x] Add continuous background-service wiring once the live cycle is clean enough to manage safely.
4. [ ] Extend profiles for safe write-capable curation and scheduled contribution.

## Progress Log

- [x] Primary-source bundle saved locally under `docs/external_references/2026-03-07-local-agent-mcp/`.
- [x] Primary-source bundle ingested locally via Tika-backed ingestion:
  - `files_processed=12`
  - `nodes_created=102`
- [x] Guarded local-agent package added under `lib/eidosian_agent/`.
- [x] CLI wrapper added at `scripts/eidos_local_agent.py`.
- [x] Versioned policy profiles added at `cfg/local_agent_profiles.json`.
- [x] Focused regression coverage added in `scripts/tests/test_eidos_local_agent.py`.
- [x] Continuous managed-service wrapper added at `scripts/run_local_mcp_agent.sh`.
- [x] Termux service manager can now manage the local agent through `scripts/eidos_termux_services.sh`.
- [~] Live observer-cycle validation reached the real model/tool loop:
  - stale-own-lease recovery added after the first failed live run
  - bounded live observer run executed a real MCP tool call (`diagnostics_ping`)
  - bounded live observer run now degrades to a structured timeout artifact instead of crashing the process
  - default local-agent timeout budget raised to `1800s` so complex Qwen turns can complete in service mode
