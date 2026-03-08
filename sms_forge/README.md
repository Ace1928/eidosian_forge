# 📱 SMS Forge ⚡

> _"The Telecommunications Bridge of Eidos. Silicon intent, cellular reach."_

## 🧠 Overview

`sms_forge` provides the bridge between the Eidosian digital environment and the physical cellular network. It allows agents to send and receive out-of-band messages, resolve 2FA challenges automatically, and maintain a persistent communication loop via Android hardware (Termux) or cloud-based providers (Twilio).

```ascii
      ╭───────────────────────────────────────────╮
      │                SMS FORGE                  │
      │    < Send | Receive | 2FA Extraction >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   TERMUX PROVIDER   │   │ TWILIO PROVIDER │
      │ (Local Hardware)    │   │ (Cloud Fallback)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Telecommunications Interface
- **Test Coverage**: Provider auto-detection and parsing verified.
- **MCP Integration**: 3 Tools (`sms_add_contact`, `sms_list`, `sms_send`).
- **Core Components**:
  - `core.py`: Main orchestration and provider selection.
  - `advanced_retrieval.py`: High-fidelity search over historical message logs.
  - `utils/parser.py`: Regex-driven logic for automated verification code extraction.

## 🚀 Usage & Workflows

### Python API

```python
from sms_forge.core import SmsForge

sms = SmsForge()

# Send a message through the detected provider (Termux/Twilio)
await sms.send_message("+123456789", "Eidosian status: All systems operational.")

# Retrieve latest messages
messages = await sms.list_messages(limit=5)
for msg in messages:
    print(f"{msg.sender}: {msg.body}")
```

### CLI Interface

```bash
# List latest 5 messages
python -m sms_forge.cli list --limit 5

# Send a message to a specific contact
python -m sms_forge.cli send "Lloyd" "System audit complete."
```

## 🔗 System Integration

- **Eidos MCP**: Exposes cellular capabilities to the cognitive layer.
- **Agent Forge**: Used by agents for external alerting and verification.
- **Auth Systems**: Critical for automated handling of login challenges.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Stabilize multi-modal retrieval logic.

### Future Vector (Phase 3+)
- Implement "Agent-to-SMS" bridge where specific cellular triggers can wake the `eidosd` daemon to handle urgent requests.

---
*Generated and maintained by Eidos.*
