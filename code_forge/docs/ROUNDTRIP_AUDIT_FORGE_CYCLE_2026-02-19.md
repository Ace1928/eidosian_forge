# Code Forge Roundtrip Validation: `audit_forge`

Date: 2026-02-19  
Cycle ID: `audit_forge_cycle04`

## Objective
- Ingest and digest `audit_forge`.
- Sync code intelligence into Knowledge Forge.
- Export GraphRAG corpus.
- Reconstruct `audit_forge` from Code Forge library.
- Verify parity and apply reconstruction transactionally.

## Command
```bash
PYTHONPATH=lib:code_forge/src ./eidosian_venv/bin/python -m code_forge.cli --json \
  roundtrip audit_forge \
  --workspace-dir data/code_forge/roundtrip/audit_forge_cycle04 \
  --mode analysis \
  --sync-knowledge \
  --export-graphrag \
  --graphrag-output-dir data/code_forge/roundtrip/audit_forge_cycle04/graphrag \
  --graph-export-limit 500 \
  --apply \
  --backup-root Backups/code_forge_roundtrip
```

## Result Summary
- `parity_pass`: `true`
- source files in scope: `19`
- reconstructed files: `19`
- hash mismatches: `0`
- missing files: `0`
- extra files: `0`
- apply mode: enabled, `noop=true` (already in sync)

## Integration Summary
- `knowledge_sync.run_id`: `b94268be05564cba`
- `knowledge_sync.scanned_units`: `118`
- `graphrag_export.exported`: `58`
- `graphrag_export.skipped`: `60`
- `graphrag_export.by_language`: `python=54`, `external=2`, `markdown=1`, `shell=1`

## Artifacts
- Roundtrip summary:
  - `data/code_forge/roundtrip/audit_forge_cycle04/roundtrip_summary.json`
- Parity report:
  - `data/code_forge/roundtrip/audit_forge_cycle04/parity_report.json`
- Reconstruction manifest:
  - `data/code_forge/roundtrip/audit_forge_cycle04/reconstructed/reconstruction_manifest.json`
- Digester outputs:
  - `data/code_forge/roundtrip/audit_forge_cycle04/digester/archive_digester_summary.json`
  - `data/code_forge/roundtrip/audit_forge_cycle04/digester/repo_index.json`
  - `data/code_forge/roundtrip/audit_forge_cycle04/digester/duplication_index.json`
  - `data/code_forge/roundtrip/audit_forge_cycle04/digester/triage.json`

## Contract Notes
- Roundtrip parity and apply operate on supported source extensions only.
- Integration exports scope to active run; when `units_created=0`, digest falls back to the latest effective run for that root so exports remain meaningful in idempotent cycles.
