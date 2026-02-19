# Eidosian Documentation Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](runtime/processor_status.json)

Doc Forge is the production documentation processor for `eidosian_forge`.

## What It Does

- Scans the repository for supported text-like files:
  - source code, configs, markdown/text, html/xml, csv/tsv, pdf, docx
- Extracts content and generates technical Markdown with a local model.
- Writes generated docs to **staging**, then runs a federated quality gate.
- Only approved docs are finalized into runtime index/final outputs.
- Resumes safely across restarts (persistent state + atomic writes).
- Exposes a live status API and modern dashboard UI.

## Runtime Paths

- Processor state: `doc_forge/runtime/processor_state.json`
- Status snapshot: `doc_forge/runtime/processor_status.json`
- Staging docs: `doc_forge/runtime/staging_docs/`
- Approved docs: `doc_forge/runtime/final_docs/`
- Rejected docs: `doc_forge/runtime/rejected_docs/`
- Judge scorecards: `doc_forge/runtime/judgments/`
- Index: `doc_forge/runtime/doc_index.json`

## Ports

Doc Forge reads defaults from `config/ports.json` when env values are unset/empty.

- Dashboard/API: `doc_forge_dashboard` (default `8930`)
- Managed local model server: `doc_forge_llm` (default `8093`)
- Full registry: `docs/PORT_REGISTRY.md`

## Start

```bash
./doc_forge/scripts/run_forge.sh
```

Open dashboard:

```bash
http://127.0.0.1:8930/
```

Key API endpoints:

- `GET /health`
- `GET /api/status`
- `GET /api/index?limit=200`
- `GET /api/recent?limit=40`

## Service Integration

Termux startup integration is handled by:

- `scripts/eidos_termux_services.sh`
- `.bashrc` service hooks (`start-shell` / `exit-shell`)

Doc Forge autostart is enabled with `EIDOS_ENABLE_DOC_FORGE_AUTOSTART=1`.

## Quality Gate

The federated gate uses independent judges for:

- heading/structure contract
- anti-placeholder checks
- source-grounding overlap
- symbol coverage from source
- specificity density

Only outputs that clear threshold and minimum per-judge floors are finalized.
