#!/usr/bin/env python3
"""
Eidos Hybrid Chat Sidecar — Server (Linux, headed Playwright)

Goal:
  Run a real, headed browser session (you log in like a human),
  and expose a local WebSocket bridge so a terminal client can:
    - send messages (type + Enter)
    - receive assistant output as it streams (delta printing)
    - control session (/new, /reset, /persist)

Why this approach:
  - No API keys, no separate billing
  - No OAuth automation (you authenticate normally)
  - The "as though from Firefox" requirement is satisfied because it *is* a browser.

Security model:
  - Binds to localhost by default (127.0.0.1)
  - WebSocket is unauthenticated; keep it local
  - Storage state (cookies/local storage) is sensitive: protect file permissions

Usage (recommended):
  python eidos_server.py --port 8928
  # then in another terminal:
  python eidos_client.py --ws ws://127.0.0.1:8928

Notes on robustness:
  - ChatGPT web UI DOM changes. All selectors are centralized in ChatDOM.
  - Message virtualization can reorder DOM; we dedupe with rolling hash window.
  - Assistant "streaming" is detected by observing the last assistant message text growing.

Dependencies:
  pip install playwright websockets
  playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import websockets
from websockets.server import WebSocketServerProtocol

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

CHAT_URL = "https://chat.openai.com/"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8928
DEFAULT_STATE_PATH = os.path.expanduser("~/.eidos_chatgpt_state.json")
DEFAULT_LOG_PATH = os.path.expanduser("~/.eidos_sidecar_server.log")

SCAN_INTERVAL_S = 0.20            # DOM scan cadence
ASSISTANT_STABLE_WINDOW_S = 0.90  # unchanged duration => "stable" marker
DEDUP_WINDOW = 800                # rolling message-hash memory (resists DOM virtualization)

WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 20
WS_MAX_SIZE = 16 * 1024 * 1024


def now() -> float:
    return time.time()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def lcp_len(a: str, b: str) -> int:
    """
    Longest common prefix length. Used for delta streaming.
    Robust even if the UI rewraps text (sometimes inserts/removes whitespace).
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def safe_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass
class DOMMessage:
    role: str   # "user" or "assistant"
    text: str


class ChatDOM:
    """
    Selector strategy container.
    When ChatGPT UI changes, update selectors here instead of hunting through logic.
    """

    # Message nodes commonly provide this attribute
    MSG_NODE_SEL = "[data-message-author-role]"
    ROLE_ATTR = "data-message-author-role"

    # Text input is commonly a textarea
    TEXTBOX_SEL = "textarea"

    # "New chat" UI changes frequently; we try multiple hooks
    NEW_CHAT_CANDIDATES = (
        "button:has-text('New chat')",
        "a:has-text('New chat')",
        "[aria-label='New chat']",
        "[data-testid='new-chat-button']",
        "button:has-text('New')",
    )

    async def wait_for_chat_ready(self, page: Page, timeout_ms: int = 180_000) -> bool:
        """
        Returns True if a textbox appears within timeout, otherwise False.
        Useful for detecting "login complete".
        """
        try:
            await page.locator(self.TEXTBOX_SEL).wait_for(timeout=timeout_ms)
            return True
        except Exception:
            return False

    async def read_messages(self, page: Page) -> List[DOMMessage]:
        """
        Snapshot messages from DOM in order.
        WARNING: DOM virtualization means not all history is always present.
        """
        out: List[DOMMessage] = []
        nodes = page.locator(self.MSG_NODE_SEL)
        count = await nodes.count()
        for i in range(count):
            node = nodes.nth(i)
            role = (await node.get_attribute(self.ROLE_ATTR) or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            text = (await node.inner_text()).strip()
            if not text:
                continue
            out.append(DOMMessage(role=role, text=text))
        return out

    async def get_last_assistant_text(self, page: Page) -> str:
        """
        Best-effort: find last assistant message and return its text.
        """
        nodes = page.locator(f"{self.MSG_NODE_SEL}[{self.ROLE_ATTR}='assistant']")
        cnt = await nodes.count()
        if cnt <= 0:
            return ""
        return (await nodes.nth(cnt - 1).inner_text()).strip()

    async def send_text(self, page: Page, text: str) -> None:
        box = page.locator(self.TEXTBOX_SEL)
        await box.wait_for(timeout=60_000)
        await box.fill(text)
        await box.press("Enter")

    async def new_chat(self, page: Page) -> bool:
        """
        Try to click "New chat" or otherwise navigate to a clean chat context.
        Returns True if something plausible happened.
        """
        for sel in self.NEW_CHAT_CANDIDATES:
            loc = page.locator(sel)
            try:
                if await loc.count():
                    await loc.first.click(timeout=1500)
                    return True
            except Exception:
                pass

        # Fallback: go to root (often lands in most recent chat; not always "new", but stable)
        try:
            await page.goto(CHAT_URL, wait_until="domcontentloaded")
            return True
        except Exception:
            return False

    async def reset(self, page: Page) -> None:
        await page.reload(wait_until="domcontentloaded")


class EidosSidecarServer:
    """
    Headed browser controller + WebSocket event bridge.

    Design:
      - A scanning loop reads DOM, pushes structured events to connected clients
      - A command loop receives client commands and manipulates the page
      - Supports multiple clients (broadcast events), but accepts commands from the most recent
        connected client by default (configurable pattern; safe enough for localhost use)
    """

    def __init__(self, host: str, port: int, state_path: str, log_path: str):
        self.host = host
        self.port = port
        self.state_path = state_path
        self.log_path = log_path

        self.dom = ChatDOM()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        self.clients: List[WebSocketServerProtocol] = []
        self.command_owner: Optional[WebSocketServerProtocol] = None

        # Message dedupe: rolling set of recent hashes
        self.recent_hashes: Deque[str] = deque(maxlen=DEDUP_WINDOW)

        # Assistant streaming state
        self.last_assistant_full: str = ""
        self.last_assistant_change_ts: float = 0.0
        self.assistant_stable_emitted_for: str = ""  # last full text we emitted stable for

        self.stop_event = asyncio.Event()

    async def _broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        msg = safe_json(payload)
        stale: List[WebSocketServerProtocol] = []
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            try:
                self.clients.remove(ws)
            except ValueError:
                pass
            if self.command_owner is ws:
                self.command_owner = None

    async def _emit_status(self, s: str) -> None:
        logging.info(s)
        await self._broadcast({"type": "status", "data": s})

    async def _emit_error(self, s: str) -> None:
        logging.error(s)
        await self._broadcast({"type": "error", "data": s})

    async def _save_state(self) -> bool:
        if not self.context:
            return False
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            await self.context.storage_state(path=self.state_path)
            os.chmod(self.state_path, 0o600)
            return True
        except Exception as e:
            await self._emit_error(f"Failed saving storage_state: {e!r}")
            return False

    async def _launch_browser(self) -> None:
        async with async_playwright() as p:
            self.browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")

            if os.path.exists(self.state_path):
                self.context = await self.browser.new_context(storage_state=self.state_path)
                logging.info("Loaded existing storage_state.")
            else:
                self.context = await self.browser.new_context()
                logging.info("No storage_state found; starting fresh context.")

            self.page = await self.context.new_page()
            await self.page.goto(CHAT_URL, wait_until="domcontentloaded")

            await self._emit_status("Headed Chromium launched. Log in normally in the browser window if needed.")

            # If already logged in, textbox appears quickly; otherwise wait quietly.
            ready = await self.dom.wait_for_chat_ready(self.page, timeout_ms=10_000)
            if ready:
                await self._emit_status("Chat UI detected (textbox present).")
            else:
                await self._emit_status("Chat UI not ready yet. Complete login in the browser window.")

            # Start WS server + scan loop concurrently inside the same Playwright lifetime
            ws_task = asyncio.create_task(self._run_ws_server())
            scan_task = asyncio.create_task(self._scan_loop())

            # Graceful shutdown signals
            def _sig_handler(*_):
                self.stop_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, _sig_handler)
                except Exception:
                    pass

            await self.stop_event.wait()

            # Shutdown procedures
            await self._emit_status("Shutting down. Attempting to save storage_state...")
            await self._save_state()

            for t in (ws_task, scan_task):
                t.cancel()
            await asyncio.gather(ws_task, scan_task, return_exceptions=True)

            try:
                if self.browser:
                    await self.browser.close()
            except Exception:
                pass

    async def _scan_loop(self) -> None:
        """
        Scan DOM periodically and emit:
          - message events (deduped via rolling hash window)
          - assistant_stream deltas (LCP-based)
          - assistant_stable when unchanged for a window
        """
        assert self.page is not None
        page = self.page

        await self._emit_status(f"DOM scanner running (interval={SCAN_INTERVAL_S:.2f}s, dedupe_window={DEDUP_WINDOW}).")

        while not self.stop_event.is_set():
            try:
                # If not logged in, avoid hammering. Just look for readiness.
                ready = await self.dom.wait_for_chat_ready(page, timeout_ms=200)
                if not ready:
                    await asyncio.sleep(0.75)
                    continue

                # Message snapshot
                messages = await self.dom.read_messages(page)
                new_events = []

                for m in messages:
                    # Hash only role+text; this dedupes repeats across virtualized DOM shifts
                    h = sha256_text(m.role + "\n" + m.text)
                    if h in self.recent_hashes:
                        continue
                    self.recent_hashes.append(h)
                    new_events.append({"role": m.role, "text": m.text, "hash": h})

                if new_events:
                    await self._broadcast({"type": "messages", "data": new_events})

                # Assistant streaming delta
                last_assistant = await self.dom.get_last_assistant_text(page)
                if last_assistant:
                    if last_assistant != self.last_assistant_full:
                        prev = self.last_assistant_full
                        cur = last_assistant

                        lcp = lcp_len(prev, cur)
                        delta = cur[lcp:]

                        # If the overlap is tiny but prev isn't tiny, it's probably a DOM replacement/reflow.
                        mode = "delta"
                        if prev and lcp < min(20, len(prev) // 5):
                            mode = "reset"
                            delta = cur  # client should replace

                        self.last_assistant_full = cur
                        self.last_assistant_change_ts = now()
                        self.assistant_stable_emitted_for = ""  # allow new stable emission

                        await self._broadcast({
                            "type": "assistant_stream",
                            "data": {
                                "mode": mode,
                                "delta": delta,
                                "full": cur,
                                "lcp": lcp,
                            }
                        })

                    else:
                        # unchanged; mark stable if enough time passed and we haven't already emitted stable for this text
                        if self.last_assistant_full and self.assistant_stable_emitted_for != self.last_assistant_full:
                            if (now() - self.last_assistant_change_ts) >= ASSISTANT_STABLE_WINDOW_S:
                                self.assistant_stable_emitted_for = self.last_assistant_full
                                await self._broadcast({
                                    "type": "assistant_stable",
                                    "data": {"full": self.last_assistant_full}
                                })

                await asyncio.sleep(SCAN_INTERVAL_S)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._emit_error(f"scan_loop error: {e!r}")
                await asyncio.sleep(0.8)

    async def _handle_ws(self, ws: WebSocketServerProtocol) -> None:
        self.clients.append(ws)
        self.command_owner = ws  # last connected wins (simple + deterministic)

        await ws.send(safe_json({"type": "status", "data": f"Connected to Eidos server on ws://{self.host}:{self.port}"}))
        await ws.send(safe_json({"type": "status", "data": f"storage_state: {self.state_path}"}))
        await ws.send(safe_json({"type": "status", "data": "Commands: send/new/reset/persist/quit"}))

        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    await ws.send(safe_json({"type": "error", "data": "Invalid JSON"}))
                    continue

                mtype = msg.get("type")
                data = msg.get("data", "")

                # Only accept commands from the command owner (prevents multi-client chaos)
                if ws is not self.command_owner and mtype in ("send", "new", "reset", "persist", "quit"):
                    await ws.send(safe_json({"type": "error", "data": "Not command owner (another client connected)."}))
                    continue

                if mtype == "send":
                    text = str(data).strip()
                    if not text:
                        continue
                    assert self.page is not None
                    try:
                        await self.dom.send_text(self.page, text)
                    except Exception as e:
                        await self._emit_error(f"send failed: {e!r}")

                elif mtype == "new":
                    assert self.page is not None
                    ok = await self.dom.new_chat(self.page)
                    await self._emit_status(f"New chat attempted: {'ok' if ok else 'failed'}")

                elif mtype == "reset":
                    assert self.page is not None
                    await self.dom.reset(self.page)
                    await self._emit_status("Page reset (reload).")

                elif mtype == "persist":
                    ok = await self._save_state()
                    await self._emit_status(f"storage_state save: {'ok' if ok else 'failed'}")

                elif mtype == "quit":
                    await self._emit_status("Client requested quit.")
                    break

                else:
                    await ws.send(safe_json({"type": "error", "data": f"Unknown command: {mtype!r}"}))

        finally:
            try:
                self.clients.remove(ws)
            except ValueError:
                pass
            if self.command_owner is ws:
                self.command_owner = self.clients[-1] if self.clients else None

    async def _run_ws_server(self) -> None:
        async def handler(ws: WebSocketServerProtocol):
            await self._handle_ws(ws)

        async with websockets.serve(
            handler,
            self.host,
            self.port,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            max_size=WS_MAX_SIZE,
        ):
            await self._emit_status(f"WebSocket server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run until cancelled

    async def run(self) -> None:
        await self._launch_browser()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eidos hybrid ChatGPT web sidecar server (headed Playwright + local WebSocket bridge)."
    )
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"WebSocket port (default {DEFAULT_PORT})")
    p.add_argument("--state", default=DEFAULT_STATE_PATH, help=f"Playwright storage_state path (default {DEFAULT_STATE_PATH})")
    p.add_argument("--log", default=DEFAULT_LOG_PATH, help=f"Log path (default {DEFAULT_LOG_PATH})")
    return p


def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.log)

    srv = EidosSidecarServer(args.host, args.port, args.state, args.log)
    try:
        asyncio.run(srv.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
