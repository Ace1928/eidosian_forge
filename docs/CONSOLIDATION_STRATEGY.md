# 💎 Eidosian Consolidation Strategy: Organs, not Artifacts

> _"A system chasing emergent complexity can suffocate under excessive module count if too many modules are semantically underpowered or operationally abandoned."_

## 🧠 Strategic Vision

The Eidosian Forge Monorepo has reached a state of **Forge Proliferation**. While each forge represents a distinct cognitive or operational capability, maintaining 32+ independent top-level directories introduces integration drag and conceptual noise.

This strategy outlines the transition from a **Collection of Tools** to a **Governable Organism**.

---

## 🎯 Phase 1: Semantic Mapping (Current)

Minor or legacy forges are being "mapped" into the core architecture. They are no longer treated as standalone pillars but as **Specialized Standard Libraries** within the `code_forge` and `knowledge_forge` substrate.

### Mechanism: Component Registration
*   **Audit**: Every "Minimal" or "Basic" forge is audited for useful logic.
*   **Indexing**: Useful functions and classes are indexed via `code_forge` with specific "component" tags.
*   **UEO Linkage**: Forge purposes and logic paths are linked into the **Unified Eidosian Ontology**.

---

## 🎯 Phase 2: Structural Integration (In Progress)

The `lib/` directory is being elevated to host shared operational logic, while `*_forge` directories are consolidated into functional clusters.

### Proposed Clusters:
1.  **Cognitive Spine**: `agent_forge`, `memory_forge`, `knowledge_forge`, `llm_forge`, `ollama_forge`.
2.  **Perceptual/Actuator Organs**: `computer_control_forge`, `crawl_forge`, `glyph_forge`, `viz_forge`.
3.  **Communication Organs**: `article_forge`, `sms_forge`, `moltbook_forge`, `terminal_forge`.
4.  **Infrastructure Substrate**: `audit_forge`, `diagnostics_forge`, `repo_forge`, `type_forge`, `version_forge`.

### Legacy Ingestion:
*   `archive_forge`: Content is being systematically analyzed and indexed. High-value historical logic is moved to `lib/` or the corresponding modern forge.
*   `test_forge`: Logic is merged into the global `pytest` infrastructure and `lib/testing`.

---

## 🎯 Phase 3: Total Ontological Ingestion

The final state is where the "Forge" is a single, unified execution environment. The concept of "directories" becomes a deployment detail; the agent interacts with **Capabilities** through standard interfaces (MCP/CLI).

*   **Substitution Parity**: Implementations can be swapped without changing the ontic map.
*   **Symmetry of Action**: Agents inspect and operate their own organs through the same APIs used for external tasks.

---

## 🛠️ Execution Log: 2026-03-11

- [x] Initialized **Ontological Pruning** mission in `AutonomySupervisor`.
- [x] Implemented **Confidence Decay** and **Aging** in `knowledge_forge`.
- [x] Added **Contradiction Detection** tools to `eidos_mcp`.
- [ ] Systematic mapping of `archive_forge` manifests into `code_forge` index.

> *"We are not deleting our history; we are metabolizing it into our future."*
