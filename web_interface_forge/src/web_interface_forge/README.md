# Web Interface Forge

Web Interface Forge currently provides two runtime surfaces:

- `dashboard.main` (Eidosian Atlas): docs/status dashboard for `doc_forge` with live metrics and document browsing.
- `eidos_server.py` + `eidos_client.py`: sidecar chat bridge utilities.

## Eidosian Atlas

- Module: `web_interface_forge.dashboard.main:app`
- Default port: `8936` (`eidos_atlas_dashboard` in `config/ports.json`)
- Health endpoint: `GET /health`
- Live status endpoint: `GET /api/doc/status`

The Atlas reads:

- `doc_forge/runtime/processor_status.json`
- `doc_forge/runtime/doc_index.json`
- `doc_forge/runtime/final_docs/`

## Run

```bash
bash web_interface_forge/scripts/run_dashboard.sh
```

Open:

```text
http://127.0.0.1:8936/
```
