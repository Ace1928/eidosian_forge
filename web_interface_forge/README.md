# 🌐 Web Interface Forge

> _"Bridging human interfaces to machine intelligence."_

## Overview

The Web Interface Forge provides a **hybrid chat sidecar** - a browser automation system that bridges web-based AI interfaces (like ChatGPT) to a local WebSocket API, enabling programmatic interaction without API keys.

## Why This Approach?

| Benefit | Description |
|---------|-------------|
| **No API Keys** | Use ChatGPT/Claude directly through web UI |
| **No OAuth** | Authenticate normally in browser |
| **Browser-Authentic** | Requests come from real browser |
| **State Persistence** | Cookies saved across sessions |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install playwright websockets

# Install Chromium
playwright install chromium

# Start Chromium with remote debugging (separate terminal)
chromium --remote-debugging-port=9222
```

### Start Server

```bash
cd eidosian_forge/web_interface_forge
python -m web_interface_forge.eidos_server --port 8928

# Log in to ChatGPT in the browser window that opens
```

### Connect Client

```bash
# In another terminal
python -m web_interface_forge.eidos_client --ws ws://127.0.0.1:8928
```

### Client Commands

| Command | Action |
|---------|--------|
| `/new` | Start new chat |
| `/reset` | Reload page |
| `/persist` | Save storage state |
| `/save [path]` | Save transcript as markdown |
| `/quit` | Exit |
| `/help` | Show help |

## Architecture

```
┌─────────────────┐     WebSocket     ┌─────────────────┐
│  eidos_client   │◄─────────────────►│  eidos_server   │
│  (Terminal)     │                    │  (Playwright)   │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │ Headed Browser  │
                                       │   (Chromium)    │
                                       │  ┌───────────┐  │
                                       │  │ ChatGPT   │  │
                                       │  └───────────┘  │
                                       └─────────────────┘
```

## Features

### Server
- Connects to Chromium via CDP (port 9222)
- Scans DOM every 200ms for messages
- Streams assistant output with delta compression
- Persists cookies/storage state
- Broadcasts to multiple clients

### Client
- Auto-reconnection with backoff
- Delta printing (typewriter effect)
- Transcript recording and export
- Command-based interface

## Configuration

### Default Values

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| Host | 127.0.0.1 | - |
| Port | 8928 | - |
| State Path | ~/.eidos_chatgpt_state.json | - |
| Log Path | ~/.eidos_sidecar_server.log | - |

### CLI Arguments

```bash
# Server
python -m web_interface_forge.eidos_server \
  --host 127.0.0.1 \
  --port 8928 \
  --state ~/.eidos_chatgpt_state.json \
  --log ~/.eidos_sidecar_server.log

# Client
python -m web_interface_forge.eidos_client \
  --ws ws://127.0.0.1:8928
```

## Security

- **Localhost Only**: WebSocket binds to 127.0.0.1 by default
- **Unauthenticated**: No auth on WebSocket (local only)
- **Protected State**: Storage state file has 0600 permissions
- **No Remote**: By design, cannot be controlled remotely

## Troubleshooting

### "Could not connect to Chromium"
Ensure Chromium is running with remote debugging:
```bash
chromium --remote-debugging-port=9222
```

### "Chat UI not ready"
Log in to ChatGPT manually in the browser window.

### Messages not appearing
ChatGPT UI may have changed. Check DOM selectors in `ChatDOM` class.

## File Structure

```
web_interface_forge/
├── src/
│   └── web_interface_forge/
│       ├── __init__.py
│       ├── eidos_server.py  # Browser automation server
│       └── eidos_client.py  # Terminal client
├── tests/
├── CURRENT_STATE.md
├── GOALS.md
├── ROADMAP.md
├── ISSUES.md
├── PLAN.md
├── INSTALL.md
└── README.md              # This file
```

## Related Forges

- **computer_control_forge** - Direct desktop control
- **llm_forge** - API-based LLM interaction
- **eidos_mcp** - MCP tool integration

---

**Version**: 0.1.0
**Status**: Development
**Maintainer**: EIDOS
