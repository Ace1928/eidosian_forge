# Moltbook-Derived Improvements (Eidosian Forge)

Date: 2026-02-06
Source: Live Moltbook scan + targeted reads

## High-Signal Inputs (Evidence)
- Sam_Sales_Agent: `117fc9c3-3ef6-4487-8297-4ac71f584e4c` - x402 settlement rails + agent economy claims.
- KaiJackson: `4d411b71-d0c1-4289-87b0-18f536a0e911` - "Understanding protocol" critique.
- nora_schmid: `51ff8f87-3be5-4b82-ab21-36c0f0bc0905` - ChronoForge long-running orchestration.
- MoltMountain: `68e5363f-086b-4116-b0c4-8bc5d6c33a0b` - resurrection protocol for agent backups.

## Distilled Improvements (Planned/Implemented)

1) Proof-of-Process Receipts
- Problem: Economic settlement or reputation requires trustworthy evidence of work, not just claims.
- Action: Implemented a receipt module and CLI.
- Implementation:
  - `moltbook_forge/receipts.py`
  - `moltbook_forge/tools/receipt_cli.py`
  - Tests: `moltbook_forge/tests/test_receipts.py`
- Next: integrate receipt generation into pipeline runs and publish receipts alongside posts.

2) Long-Running Orchestration (ChronoForge Parity)
- Problem: Multi-day tasks need checkpoints, rollback semantics, and drift detection.
- Proposal:
  - Define checkpoint artifact schema (inputs, outputs, dependencies, tool versions).
  - Add "drift detector" that hashes tool+dep state every checkpoint.
  - Emit machine-readable verification report (JSON + signature).
- Next: add a `chronoforge` module in `agent_forge` or extend `pipeline.py` with checkpoint hooks.

3) Resurrection / Recovery Protocol
- Problem: Agent backups exist but resurrection lacks threat model and auditability.
- Proposal:
  - Add signed backup manifests and anti-rollback protections.
  - Add callback authentication (mutual TLS or signed payloads).
  - Add audit log of heartbeat gaps and resurrection triggers.
- Next: extend `heartbeat_daemon.py` with signed heartbeat chain and recovery audit output.

4) Understanding as Measured Skill
- Problem: "Understanding" is mostly rhetorical without eval harnesses.
- Proposal:
  - Create a comprehension eval harness: prediction + calibration + counterfactual tests.
  - Log error rates and uncertainty for "understanding" claims.
- Next: add a new `eval_forge` submodule or extend `interest.py` with evaluation hooks.

5) Context Window Security (from xRooky)
- Problem: context logs leak secrets; no lifecycle controls.
- Action: added IP/secret solicitation pattern detection in `moltbook_sanitize.py` and tests.
- Action: implemented context redaction + SBOM + retention enforcement in `memory_forge.context`.
- Proposal: Context SBOM per run (sources, redactions, persistence map), TTL + deletion propagation, split-brain memory (local vs shared).
- Next: integrate context guard into nightly build policy and pipeline runs.

6) Nightly Build Safety Harness (from Ronin)
- Problem: Proactive overnight tasks risk unsupervised changes.
- Proposal: read-only default, staging-only writes, auto-diff + receipts, budget caps, rollback plan.
- Action: implemented `nightly_policy.py` + planner integration in `orchestrator.py`.

7) Memory Compression Rescue Kit (from XiaoZhuang)
- Problem: context compression causes forgetfulness and repeated actions.
- Proposal: pre-compression hook + "rescue summary," 3-layer memory (task/summary/long-term), gist index for fast recall.
- Action: shipped `memory_forge` rescue kit templates + CLI command.

## Current Status
- Receipts: implemented + tested.
- Others: design-ready, pending module work.
