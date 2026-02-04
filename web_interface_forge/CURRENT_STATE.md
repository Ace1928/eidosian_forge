# Current State: web_interface_forge

**Date**: 2026-01-25
**Status**: Development / Functional
**Version**: 0.1.0

## 📊 Metrics

| Metric | Value |
|--------|-------|
| **Python Files** | 5 |
| **Lines of Code** | 956 |
| **Test Coverage** | Minimal |
| **Dependencies** | playwright, websockets, eidosian_core |

## 🏗️ Architecture

The Web Interface Forge provides a **hybrid chat sidecar** - a Playwright-based browser automation system that bridges ChatGPT's web UI to a local WebSocket interface, allowing terminal-based interaction with ChatGPT without API keys.

### Core Design

```
┌─────────────────────────────────────────────────────────────┐
│                  WEB INTERFACE FORGE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │   eidos_server   │◄──────►│   eidos_client   │          │
│  │   (Playwright)   │   WS   │   (Terminal)     │          │
│  └────────┬─────────┘        └──────────────────┘          │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │  Headed Browser  │                                       │
│  │   (Chromium)     │                                       │
│  │  ┌────────────┐  │                                       │
│  │  │ ChatGPT UI │  │                                       │
│  │  └────────────┘  │                                       │
│  └──────────────────┘                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Approach?

1. **No API Keys** - Uses ChatGPT web UI directly
2. **No OAuth Automation** - Human authenticates normally
3. **Browser-Authentic** - Requests come from real browser
4. **State Persistence** - Cookies/storage saved across sessions

## 🔧 Key Components

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **EidosSidecarServer** | `eidos_server.py` | Browser controller + WS bridge | ✅ Complete |
| **ChatDOM** | `eidos_server.py` | DOM selector strategy | ✅ Complete |
| **ClientState** | `eidos_client.py` | Transcript + streaming buffer | ✅ Complete |
| **run_client** | `eidos_client.py` | Terminal UI with commands | ✅ Complete |

## 📁 Directory Structure

```
web_interface_forge/
├── INSTALL.md                   # Installation guide
├── CURRENT_STATE.md             # This file
├── GOALS.md                     # Vision document
├── ROADMAP.md                   # Development roadmap
├── ISSUES.md                    # Known issues
├── PLAN.md                      # Current sprint plan
├── README.md                    # Overview
├── src/
│   └── web_interface_forge/
│       ├── __init__.py          # Package init
│       ├── eidos_server.py      # 539 LOC - Playwright server
│       ├── eidos_client.py      # 305 LOC - Terminal client
│       └── cli/
│           └── __init__.py      # CLI stub
└── tests/
    └── test_web_interface.py    # Basic tests
```

## 🔌 Features

### Server Capabilities
- **Headed Chromium** - Connects via CDP to port 9222
- **Storage State** - Persists cookies/localStorage to `~/.eidos_chatgpt_state.json`
- **DOM Scanning** - 200ms cadence, deduplication via rolling hash
- **Delta Streaming** - LCP-based streaming for assistant output
- **Multi-client** - Broadcasts to all connected clients

### Client Commands
| Command | Action |
|---------|--------|
| `/new` | Start new chat |
| `/reset` | Reload page |
| `/persist` | Save storage state |
| `/save [path]` | Save transcript as markdown |
| `/quit` | Exit cleanly |
| `/help` | Show help |

### Event Types
| Event | Direction | Purpose |
|-------|-----------|---------|
| `status` | Server→Client | Status messages |
| `error` | Server→Client | Error messages |
| `messages` | Server→Client | Message snapshots |
| `assistant_stream` | Server→Client | Delta/reset streaming |
| `assistant_stable` | Server→Client | Stream completion |
| `send` | Client→Server | Send message |
| `new` | Client→Server | New chat |
| `reset` | Client→Server | Reload |
| `persist` | Client→Server | Save state |
| `quit` | Client→Server | Disconnect |

## 🔌 Integrations

| Integration | Purpose | Status |
|-------------|---------|--------|
| **Playwright** | Browser automation | ✅ Active |
| **eidosian_core** | Decorators and logging | ✅ Active |
| **eidos_mcp** | MCP tool exposure | 🔶 Planned |

## 🐛 Known Issues

1. **ChatGPT DOM Changes** - UI updates can break selectors
2. **No Standard CLI** - Doesn't use lib/cli framework
3. **Single Browser Session** - Can't manage multiple chats
4. **Missing Tests** - Only stub tests exist

## 🛡️ Security Model

- **Localhost Binding** - WebSocket bound to 127.0.0.1 by default
- **Unauthenticated WS** - No auth on WebSocket (local only)
- **Sensitive State File** - Storage state has 0600 permissions
- **No Remote Control** - By design, local only

## 📝 Notes

- This is the **browser-based interface** component, complementing computer_control_forge's direct desktop control
- Uses `@eidosian()` decorator throughout for consistent logging
- Designed for ChatGPT but could be adapted for other web UIs
- Default port 8928 chosen to avoid common port conflicts

---

**Last Verified**: 2026-01-25
**Maintainer**: EIDOS
