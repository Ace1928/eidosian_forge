# Runtime Artifact Audit

- Repo root: `/data/data/com.termux/files/home/eidosian_forge`
- Policy path: `/data/data/com.termux/files/home/eidosian_forge/cfg/runtime_artifact_policy.json`
- Tracked violation count: `7`
- Live generated count: `32`

## Tracked Generated Files

- `data/code_forge/vectors/index.bin`
- `data/runtime/eidos_scheduler_status.json`
- `data/runtime/entity_proof_status.json`
- `data/runtime/forge_coordinator_status.json`
- `data/runtime/local_mcp_agent/history.jsonl`
- `data/runtime/test_autonomy/ledger/continuity_ledger.jsonl`
- `data/tiered_memory/vectors/index.bin`

## Live Generated Files

- `data/.kb.json.8xd2eo1y.tmp`
- `data/code_forge/vectors/index.bin`
- `data/runtime/directory_docs_history.json`
- `data/runtime/directory_docs_status.json`
- `data/runtime/docs_upsert_batch_history.jsonl`
- `data/runtime/docs_upsert_batch_status.json`
- `data/runtime/eidos_scheduler_history.jsonl`
- `data/runtime/eidos_scheduler_status.json`
- `data/runtime/entity_proof_status.json`
- `data/runtime/external_benchmarks`
- `data/runtime/external_benchmarks/agencybench/scenario2/20260320_072526/attempts.jsonl`
- `data/runtime/external_benchmarks/agencybench/scenario2/20260320_072526/local_mcp_agent/history.jsonl`
- `data/runtime/external_benchmarks/agencybench/scenario2/20260320_101430/attempts.jsonl`
- `data/runtime/external_sources`
- `data/runtime/forge_coordinator_status.json`
- `data/runtime/living_pipeline_history.jsonl`
- `data/runtime/living_pipeline_status.json`
- `data/runtime/local_mcp_agent/history.jsonl`
- `data/runtime/proof_refresh_history.jsonl`
- `data/runtime/proof_refresh_status.json`
- `data/runtime/qwenchat/history.jsonl`
- `data/runtime/runtime_artifact_audit_history.jsonl`
- `data/runtime/runtime_artifact_audit_status.json`
- `data/runtime/runtime_benchmark_run_history.jsonl`
- `data/runtime/runtime_benchmark_run_status.json`
- `data/runtime/session_bridge/events.jsonl`
- `data/runtime/test_autonomy/ledger/continuity_ledger.jsonl`
- `data/runtime/tmp/push_worktree/data/code_forge/vectors/index.bin`
- `data/runtime/tmp/push_worktree/data/runtime/local_mcp_agent/history.jsonl`
- `data/runtime/tmp/push_worktree/data/runtime/test_autonomy/ledger/continuity_ledger.jsonl`
- `data/runtime/tmp/push_worktree/data/tiered_memory/vectors/index.bin`
- `data/tiered_memory/vectors/index.bin`

## Recommendations

1. Untrack generated runtime artifacts or move them into ignored runtime/report directories.
1. Keep runtime writers confined to ignored paths and preserve only curated reports under reports/.
