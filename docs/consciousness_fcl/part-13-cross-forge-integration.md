# Part 13: Cross-Forge Memory/Knowledge Integration

## Goal

Connect the consciousness runtime directly to `memory_forge` and `knowledge_forge` so recall/context are produced in-band as events, candidates, and MCP-visible runtime state.

## Implemented In This Pass

1. Memory bridge module (`memory_bridge`)
- File: `agent_forge/src/agent_forge/consciousness/modules/memory_bridge.py`
- Adds defensive lazy import for `memory_forge` in Termux/Linux path layouts.
- Emits:
  - `mem.recall`
  - `mem.introspection`
  - `memory_bridge.status`
- Broadcasts:
  - `MEMORY`
  - `MEMORY_META`
- Metrics:
  - `consciousness.memory_bridge.available`
  - `consciousness.memory_bridge.recalls`
  - `consciousness.memory_bridge.insight_count`

2. Knowledge bridge module (`knowledge_bridge`)
- File: `agent_forge/src/agent_forge/consciousness/modules/knowledge_bridge.py`
- Uses `KnowledgeMemoryBridge` when available with fallback-safe import path handling.
- Emits:
  - `knowledge.context`
  - `knowledge.recall`
  - `knowledge_bridge.status`
- Broadcasts:
  - `KNOWLEDGE`
- Metrics:
  - `consciousness.knowledge_bridge.available`
  - `consciousness.knowledge_bridge.total_hits`
  - `consciousness.knowledge_bridge.memory_hits`
  - `consciousness.knowledge_bridge.knowledge_hits`

3. Kernel/runtime wiring
- Files:
  - `agent_forge/src/agent_forge/consciousness/kernel.py`
  - `agent_forge/src/agent_forge/consciousness/types.py`
  - `agent_forge/src/agent_forge/consciousness/modules/__init__.py`
  - `agent_forge/src/agent_forge/consciousness/modules/attention.py`
  - `agent_forge/src/agent_forge/consciousness/modules/sense.py`
- Default module order now includes bridge modules before attention competition.
- Added default config keys and cadence values for bridge modules.

4. Observability integration
- Files:
  - `agent_forge/src/agent_forge/consciousness/trials.py`
  - `agent_forge/src/agent_forge/consciousness/benchmarks.py`
  - `agent_forge/src/agent_forge/core/self_model.py`
- Runtime status now exposes bridge availability, queries, and hit volumes.
- Benchmark capability score now incorporates bridge activity normalization.
- Self snapshot includes bridge integration block.

5. MCP exposure
- File: `eidos_mcp/src/eidos_mcp/routers/consciousness.py`
- Added tool:
  - `consciousness_bridge_status`
- Added resource:
  - `eidos://consciousness/runtime-integrations`

6. Validation coverage
- Files:
  - `agent_forge/tests/test_consciousness_memory_knowledge_bridge.py`
  - `eidos_mcp/tests/test_mcp_tool_calls_individual.py`
  - `scripts/linux_parity_smoke.sh`
- New tests validate bridge event emission, broadcasts, kernel wiring, snapshot integration, and MCP availability.

## Remaining Work

1. World model predictive coding v1
- Replace event-type transition-only model with feature-space latent prediction and calibrated uncertainty.

2. Ablation benchmark suite
- Add first-class ablation matrix assertions for module contribution deltas.

3. Simulation/dreaming stream
- Add explicit simulated percept stream and mode-aware reporting integrity checks.
