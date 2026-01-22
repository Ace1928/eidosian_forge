# Eidos MCP Server

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Central Nervous System.**

## 🧠 Overview

`eidos_mcp` is the implementation of the [Model Context Protocol](https://github.com/model-context-protocol/mcp) server.
It exposes the capabilities of all other Forges (Memory, Knowledge, Coding, etc.) as **Tools** and **Resources** to the LLM.

## 🔗 Integrations
- **Memory**: `memory_add`, `memory_retrieve`
- **Knowledge**: `kb_add`, `grag_query`
- **System**: `run_shell_command`, `file_read`, `file_write`
- **Audit**: `audit_mark_reviewed`, `audit_add_todo`

## 🚀 Usage

```bash
# Start the server (std/stdio mode)
python -m eidos_mcp.eidos_mcp_server
```

## 🛠️ Configuration
Configuration is loaded from the **GIS** (Global Info System) or `config.json`.

## ♻️ Rollback & Change Log
- Changelog: `CHANGELOG.md`
- Snapshot: `scripts/mcp_snapshot.py snapshot --label "before-change" --log`
- Restore: `scripts/mcp_snapshot.py restore /path/to/snapshot.tar.gz --yes`

Snapshots are written to `~/.eidosian/backups/eidos_mcp` by default. Override with
`EIDOS_MCP_BACKUP_DIR`.

## 🧯 Transactional Tooling
Destructive tools (file/memory/knowledge/GIS/type/audit) run through transaction
snapshots that are idempotent and rollbackable. Transactions live under
`~/.eidosian/transactions` (override with `EIDOS_TXN_DIR`).

Common helpers:
- `transaction_list` to view recent transactions.
- `transaction_restore` to rollback by transaction id.
- Domain restores: `file_restore`, `memory_restore`, `memory_restore_semantic`,
  `kb_restore`, `gis_restore`, `type_restore_snapshot`.

`run_shell_command` defaults to safe mode. For potentially destructive commands,
provide `transaction_paths` and optional `verify_command`/`idempotency_key` to
enable automatic rollback on failure.

## 🧷 Persistent Service (systemd)
The persistent service runs the Nexus over StreamableHTTP and auto-rolls back if
an edit fails to start cleanly. A successful start promotes the new snapshot to
`last_good` and only then updates the service state.

1) Install the user service:
```bash
mkdir -p ~/.config/systemd/user
cp /home/lloyd/eidosian_forge/eidos_mcp/systemd/eidos-mcp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now eidos-mcp.service
```

2) Optional: keep it running after logout:
```bash
loginctl enable-linger lloyd
```

3) Health check (StreamableHTTP):
```bash
curl -s http://127.0.0.1:8765/mcp | head -n 1
```

Service guard state lives at `~/.eidosian/run/mcp_service_state.json` and tracks
the last known-good snapshot for automatic rollback.
