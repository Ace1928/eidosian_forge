# 📱 SMS Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Telecommunications Bridge of Eidos.**

> _"Connectivity beyond the digital fabric."_

## 📱 Overview

`sms_forge` enables Eidosian agents to interact with the global cellular network. It provides a unified abstraction for sending and receiving SMS messages, handling 2FA challenges, and providing asynchronous notification channels.

## 🏗️ Architecture (Planned)

- **Twilio Provider**: Cloud-based SMS gateway.
- **ADB Bridge**: (Planned) Direct interaction with local Android hardware for low-cost/offline messaging.
- **2FA Handler**: Specialized parser for extracting verification codes from incoming messages.

## 🔗 System Integration

- **Eidos MCP**: Exposes `sms_send` and `sms_listen` tools.
- **Agent Forge**: Agents use this for out-of-band alerting and identity verification.

## 🚀 Status

**Planning Phase**. Implementation of the Twilio client is scheduled for the next major milestone.
