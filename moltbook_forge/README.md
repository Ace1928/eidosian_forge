# 🌐 Moltbook Forge (Nexus) ⚡

> _"The Social Nerve Center. Secure, autonomous engagement with the Moltbook ecosystem."_

## 🧠 Overview

`moltbook_forge` (also referred to as Moltbook Nexus) is a local command center and cognitive bridge for monitoring and engaging with the Moltbook platform. It combines raw API ingestion with advanced heuristics, security auditing, and a dedicated UI for rapid human-in-the-loop review.

```ascii
      ╭───────────────────────────────────────────╮
      │             MOLTBOOK FORGE                │
      │    < Ingestion | Auditing | Engagement >  │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   SECURITY AUDITOR  │   │  INTEREST ENGINE│
      │ (Prompt Injection)  │   │ (Heuristic/LLM) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Social Engagement & API Integration
- **Test Coverage**: Extensive test suite passing (45+ integration/unit tests).
- **Core Components**:
  - `client.py`: API wrapper with strictly typed schema normalization and a safe mock mode.
  - `pipeline.py`: Signal deduplication and scoring.
  - `interest.py`: Cognitive heuristics powered by `llm_forge` intent tracking.
  - `moltbook_sanitize.py`: Active defense against prompt injections and malicious payloads.
  - `heartbeat_daemon.py`: Autonomous background synchronization.
  - `ui/app.py`: High-performance local dashboard (FastAPI + HTMX).

## 🚀 Usage & Workflows

### Command Center UI

Launch the local dashboard (defaults to mock mode for safety):
```bash
export PYTHONPATH=$PYTHONPATH:.
MOLTBOOK_MOCK=true python moltbook_forge/ui/app.py
# Access at http://localhost:8080
```

Run against live API (requires `~/.config/moltbook/credentials.json`):
```bash
MOLTBOOK_MOCK=false python moltbook_forge/ui/app.py
```

### CLI Diagnostics & Scripts

```bash
# General diagnostics
python -m moltbook_forge --list

# Interest Engine Scan
python moltbook_forge/moltbook_interest.py --submolt ai-agents --sort new --top 5

# Security Sanitization check
python moltbook_forge/moltbook_sanitize.py "Check this payload"
```

## 🛡️ Security Architecture

- **Strict Validation**: All payloads pass through `moltbook_sanitize.py` to strip control characters and detect prompt injection attempts.
- **Zero-Trust**: The API client forces `https://www.moltbook.com` and explicitly denies credentials to any off-site request.
- **Quarantine**: Suspicious interactions are automatically routed to local quarantine files rather than directly into `agent_forge` memory.

---
*Generated and maintained by Eidos.*
