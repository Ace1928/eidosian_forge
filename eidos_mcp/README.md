# Eidos MCP Server

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Central Nervous System of the Eidosian Forge.**

## 🧠 Overview

`eidos_mcp` is the implementation of the [Model Context Protocol](https://github.com/model-context-protocol/mcp) server.
It exposes the capabilities of all other Forges (Memory, Knowledge, Coding, etc.) as **Tools** and **Resources** to the LLM.

**Current Status**: 109 MCP tools across 20 categories, 4 plugins loaded.

## 🔗 System Integration

The Nexus integrates deeply with the Eidosian ecosystem:
- **Documentation Forge**: `eidos_mcp` is recursively documented by the `doc_forge`, ensuring its capabilities are always indexed.
- **Agent Forge**: Provides the tool registry for autonomous agents.
- **Memory Forge**: Centralizes episodic and semantic recall.
- **Global Info System (GIS)**: Acts as the configuration backbone.

## 🔗 Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| memory | 13 | Episodic/semantic memory operations |
| wf (word_forge) | 9 | Semantic lexicon and graph operations |
| tika | 8 | Document extraction via Tika |
| file | 7 | File system operations |
| kb | 7 | Knowledge base CRUD |
| eidos | 6 | Tiered memory (self/user/context) |
| tiered | 5 | Memory tier management |
| gis | 4 | Global Info System |
| type | 4 | Type registry |
| grag | 3 | GraphRAG queries |
| audit | 2 | Code audit tracking |
| diagnostics | 2 | System health |
| run | 2 | Shell/test execution |
| transaction | 2 | Transaction rollback |
| agent | 1 | Agent task execution |
| mcp | 1 | Self-upgrade |
| refactor | 1 | Code analysis |
| system | 1 | System info |
| venv | 1 | Virtual env commands |
| moltbook | 22 | Moltbook social network operations |

## 🔌 Plugins (33 additional tools)

- **self_exploration** (4 tools): Introspection, identity, provenance
- **task_automation** (6 tools): Task queue management
- **computer_control** (15 tools): Mouse, keyboard, screenshot, OCR
- **web_tools** (8 tools): Web fetch, parse, cache

## 🚀 Usage

**Recommended:** Use the portable launcher script.

```bash
# Start the server (StreamableHTTP on port 8928)
./run_server.sh
```

**Manual Start:**

```bash
# Activate the dedicated venv
source ../eidosian_venv/bin/activate

# Ensure local core package is installed
pip install -e ../lib

# Start the server (stdio)
python -m eidos_mcp.eidos_mcp_server

# Start the server (SSE/StreamableHTTP)
export EIDOS_MCP_TRANSPORT=streamable-http 
export EIDOS_MCP_MOUNT_PATH=/mcp 
export FASTMCP_PORT=8928
python -m eidos_mcp.eidos_mcp_server
```

## 🛠️ Configuration
Configuration is loaded from the **GIS** (Global Info System) or `config.json`.
Key env vars (all default-safe):
- `EIDOS_MCP_TRANSPORT` (`stdio`|`sse`|`streamable-http`)
- `EIDOS_MCP_MOUNT_PATH` (default `/mcp`, StreamableHTTP endpoint path)
- `EIDOS_MCP_STATELESS_HTTP` (`1`/`0`, default `0`) to disable session pinning and
  allow per-request handling for clients that cannot preserve MCP session state
- `EIDOS_MCP_ENABLE_COMPAT_HEADERS` (`1`/`0`, default `1`) to normalize missing
  `Accept` / `Content-Type` headers from fragile clients
- `EIDOS_MCP_ENABLE_SESSION_RECOVERY` (`1`/`0`, default `1`) to clear stale
  `mcp-session-id` headers and transparently establish a fresh session
- `EIDOS_MCP_ENABLE_ERROR_RESPONSE_COMPAT` (`1`/`0`, default `1`) to coerce
  non-MCP transport error bodies into explicit JSON with `Content-Type`
- `FASTMCP_HOST` / `FASTMCP_PORT`
- `EIDOS_FORGE_DIR`
- `EIDOS_OAUTH2_PROVIDER` (`google`|empty), `EIDOS_OAUTH2_AUDIENCE`, `EIDOS_OAUTH2_STATIC_BEARER`

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

3) Health checks (StreamableHTTP):
```bash
curl -s http://127.0.0.1:8928/health | jq .
curl -s -D - -o /dev/null http://127.0.0.1:8928/mcp
```

Service guard state lives at `~/.eidosian/run/mcp_service_state.json` and tracks
the last known-good snapshot for automatic rollback.
