# Eidos Hybrid Chat Sidecar (Linux)

This project gives you a **terminal chat interface** to your existing ChatGPT web account,
without using an API key, by running a **real headed browser** (Playwright Chromium) and
bridging it to a **local WebSocket**.

## Why this exists
- You want "chat in terminal"
- You do **not** want API keys or separate billing
- You want it to behave like the web app because it *is* the web app

## Components
- `eidos_server.py` — launches headed Chromium, loads/saves session, scans DOM, streams deltas
- `eidos_client.py` — terminal UI, reconnects automatically, supports commands

Default WebSocket:
- `ws://127.0.0.1:8928`
