# 📱 SMS Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-green.svg)](README.md)

**The Telecommunications Bridge of Eidos.**

> _"Connectivity beyond the digital fabric. Local hardware, global reach."_

## 📱 Overview

`sms_forge` is a multi-backend telecommunications bridge. It allows Eidosian agents to interact with the cellular network, enabling out-of-band messaging, 2FA challenge resolution, and asynchronous alerting.

## 🏗️ Architecture

- **Provider Pattern**: Supports multiple backends through a unified interface.
- **Termux Backend (`providers/termux.py`)**: Uses `termux-api` to leverage physical device hardware.
- **Twilio Backend (`providers/twilio.py`)**: Cloud-based fallback for Linux/Non-Termux environments.
- **Intelligent Parser (`utils/parser.py`)**: Regex-based extraction of 2FA and verification codes.

## 🔗 System Integration

- **Eidos MCP**: Exposes `sms_send`, `sms_list`, and `sms_get_codes` tools.
- **Agent Forge**: Agents use these tools for identity verification and notification.

## 🚀 Usage

### Python API

```python
from sms_forge.core import SmsForge

sms = SmsForge()

# Send a message
await sms.send_message("+123456789", "Hello from Eidos")

# Extract 2FA code from recent messages
code = await sms.get_2fa_code(sender_pattern="Google")
print(f"Your code: {code}")
```

## 🛠️ Configuration

- **Termux**: Requires `termux-api` package installed on the Android host.
- **Twilio**: Requires `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_PHONE_NUMBER` in GIS or Environment.

## 🧪 Testing

```bash
# Run the test suite
pytest sms_forge/tests/
```
