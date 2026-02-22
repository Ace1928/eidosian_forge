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

## ✅ Consciousness Validation Tools

RAC-AP construct validation outputs from `agent_forge` are exposed via:
- `consciousness_construct_validate` (run validator and return report)
- `consciousness_construct_latest` (latest persisted validator report)
- `eidos://consciousness/construct-latest` (resource snapshot)

## 🚀 Usage

**Recommended:** Use the portable launcher script.

```bash
# Start the server (default port from config/ports.json -> eidos_mcp)
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
Configuration is loaded from the **GIS** (Global Info System) first, with environment
variables overriding GIS values at runtime.
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
- `EIDOS_MCP_RATE_LIMIT_GLOBAL_PER_MIN` (default `600`) global tool-call ceiling per 60s window
- `EIDOS_MCP_RATE_LIMIT_PER_TOOL_PER_MIN` (default `300`) per-tool call ceiling per 60s window
- `EIDOS_MCP_ENFORCE_ORIGIN` (`1`/`0`, default `1`) to reject non-local browser origins
- `EIDOS_MCP_ALLOWED_ORIGINS` (comma-separated origins, default local loopback origins)
- `FASTMCP_HOST` / `FASTMCP_PORT`
- `EIDOS_FORGE_DIR`
- `EIDOS_GRAPHRAG_ROOT` (optional GraphRAG workspace override; default `graphrag_workspace`, fallback `graphrag`)
- `EIDOS_GRAPHRAG_TIMEOUT_SEC` (GraphRAG subprocess timeout used by knowledge router; default `900`)
- `EIDOS_OAUTH2_PROVIDER` (`google`|empty), `EIDOS_OAUTH2_AUDIENCE`, `EIDOS_OAUTH2_STATIC_BEARER`

GIS keys mirror these settings under the `mcp.*` namespace, for example:
- `mcp.transport`, `mcp.mount_path`, `mcp.stateless_http`
- `mcp.host`, `mcp.port`, `mcp.log_level`, `mcp.reload`
- `mcp.allowed_origins`, `mcp.enable_compat_headers`, `mcp.enable_session_recovery`
- `mcp.rate_limit_global_per_min`, `mcp.rate_limit_per_tool_per_min`
- `mcp.oauth.provider`, `mcp.oauth.audience`, `mcp.oauth.static_bearer`

`run_server.sh` defaults `EIDOS_MCP_STATELESS_HTTP=1` for Codex/Gemini compatibility.
If `FASTMCP_PORT`/`EIDOS_MCP_PORT` are unset or empty, the server falls back to
`config/ports.json` (`services.eidos_mcp.port`).

### Client Wiring (Codex + Gemini)

Codex (`~/.codex/config.toml`):
```toml
[mcp_servers.eidosian_nexus]
url = "http://127.0.0.1:8928/mcp"
```

Gemini (`~/.gemini/settings.json`):
```json
{
  "mcpServers": {
    "eidosian_nexus": {
      "httpUrl": "http://127.0.0.1:8928/mcp",
      "url": "http://127.0.0.1:8928/mcp",
      "trust": true
    }
  }
}
```

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

## 📱 Termux Service Lifecycle

Use the Termux-safe service manager:

```bash
./scripts/eidos_termux_services.sh start
./scripts/eidos_termux_services.sh status
./scripts/eidos_termux_services.sh stop
```

`~/.bashrc` should call:

```bash
/data/data/com.termux/files/home/eidosian_forge/scripts/eidos_termux_services.sh start-shell
```

and on shell exit:

```bash
/data/data/com.termux/files/home/eidosian_forge/scripts/eidos_termux_services.sh exit-shell
```

This keeps startup idempotent, prevents duplicate MCP instances, and safely shuts down managed services when the last interactive shell exits.
