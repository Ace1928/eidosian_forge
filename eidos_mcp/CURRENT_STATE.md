# Current State: eidos_mcp

**Date**: 2026-01-25
**Status**: Production / Critical Infrastructure
**Version**: 1.0.0

## 📊 Metrics

| Metric | Value |
|--------|-------|
| **Routers** | 16 |
| **Tools Exposed** | 90+ |
| **Forges Integrated** | 10+ |
| **Transports** | Stdio, SSE, StreamableHTTP |

## 🏗️ Architecture

Eidos MCP is the **Model Context Protocol server** - the central hub that exposes all forge capabilities as MCP tools for AI agents.

### Core Design

```
┌─────────────────────────────────────────────────────────────┐
│                      EIDOS MCP                               │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │              FastMCP Server (core.py)                  │  │
│  │  @tool() decorator → JSON-RPC 2.0 → MCP Protocol      │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────┴─────────────────────────────┐  │
│  │                      ROUTERS                           │  │
│  ├─────────┬─────────┬─────────┬─────────┬─────────────┤  │
│  │ memory  │knowledge│ system  │  audit  │  tika       │  │
│  ├─────────┼─────────┼─────────┼─────────┼─────────────┤  │
│  │  gis    │  types  │  nexus  │  auth   │ diagnostics │  │
│  ├─────────┼─────────┼─────────┼─────────┼─────────────┤  │
│  │refactor │tiered_  │word_    │ plugins │  code       │  │
│  │         │memory   │forge    │         │             │  │
│  └─────────┴─────────┴─────────┴─────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Routers

| Router | Tools | Purpose |
|--------|-------|---------|
| **tiered_memory** | 19 | EIDOS identity, lessons, recall |
| **knowledge** | 14 | KB facts, RAG query, context |
| **system** | 12 | File ops, shell, transactions |
| **memory** | 8 | Memory CRUD, stats |
| **tika** | 8 | Document extraction |
| **gis** | 5 | Global state persistence |
| **audit** | 5 | TODO tracking, review |
| **diagnostics** | 4 | Health, metrics |
| **word_forge** | 6 | Term management, paths |

## 🔌 Transports

| Transport | Config | Use Case |
|-----------|--------|----------|
| **Stdio** | Default | Gemini CLI, local |
| **SSE** | Port 8928 | Remote clients |
| **StreamableHTTP** | /streamable-http | Web integration |

## 🔌 Forge Integrations

- gis_forge, audit_forge, type_forge
- llm_forge, agent_forge, refactor_forge
- memory_forge, diagnostics_forge, file_forge
- knowledge_forge, word_forge

## 🛡️ Configuration

```bash
# Environment variables
EIDOS_FORGE_DIR=/home/lloyd/eidosian_forge
EIDOS_MCP_TRANSPORT=stdio|sse|streamable-http
FASTMCP_HOST=127.0.0.1
FASTMCP_PORT=8928
```

## 🐛 Known Issues

- Stdio requires clean stdout (no logging noise)
- Some forges need explicit setup

---

**Last Verified**: 2026-01-25
