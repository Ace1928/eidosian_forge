#!/usr/bin/env python3
"""
Eidos Hybrid Chat Sidecar — Client (Linux terminal)

Connects to the local Eidos server via WebSocket and provides:
  - Delta printing: assistant output appears as it "types"
  - Commands:
      /new        Start a new chat (best-effort; UI dependent)
      /reset      Reload the page
      /persist    Save Playwright storage_state (cookies/local storage)
      /save [p]   Save transcript to Markdown (default ./transcript.md)
      /quit       Exit client cleanly
      /help       Show command help

Robustness:
  - Auto-reconnect if the server restarts
  - Avoids duplicate printing:
      - "messages" events record transcript; assistant is printed mainly via stream
      - stable events finalize the assistant stream into transcript exactly once
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import websockets

DEFAULT_WS = "ws://127.0.0.1:8928"
RECONNECT_DELAY_S = 1.0
MAX_RECONNECT_DELAY_S = 10.0


@dataclass
class Entry:
    ts: float
    role: str
    text: str
    hash: str = ""


def ts_str(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def save_markdown(entries: List[Entry], path: str) -> None:
    lines = []
    lines.append("# Eidos Transcript\n\n")
    lines.append(f"_Saved: {ts_str(time.time())}_\n\n")

    for e in entries:
        who = "You" if e.role == "user" else "Eidos"
        lines.append(f"## {who} — {ts_str(e.ts)}\n\n")
        lines.append(e.text.rstrip() + "\n\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


class ClientState:
    """
    Holds transcript + streaming buffer.
    """
    def __init__(self):
        self.entries: List[Entry] = []
        self.seen_hashes: set[str] = set()

        # streaming assistant
        self.stream_open: bool = False
        self.stream_buf: str = ""
        self.stream_last_full: str = ""  # last full from server (for resets)
        self.stream_finalized_for: str = ""  # avoid double-finalize

    def _open_stream(self):
        if not self.stream_open:
            self.stream_open = True
            self.stream_buf = ""
            sys.stdout.write("\neidos> ")
            sys.stdout.flush()

    def _close_stream(self):
        if self.stream_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.stream_open = False

    def record_message(self, role: str, text: str, h: str = ""):
        if h and h in self.seen_hashes:
            return
        if h:
            self.seen_hashes.add(h)
        self.entries.append(Entry(ts=time.time(), role=role, text=text, hash=h))

    def print_user(self, text: str):
        self._close_stream()
        print(f"\nyou> {text}\n", flush=True)

    def print_status(self, text: str):
        self._close_stream()
        print(f"[status] {text}", flush=True)

    def print_error(self, text: str):
        self._close_stream()
        print(f"[error] {text}", flush=True)

    def handle_messages(self, msgs: list[dict]):
        """
        messages[] are "completed" snapshots from DOM scan.
        We store all, but we print:
          - user messages immediately
          - assistant messages ONLY if we're not actively streaming
        """
        for m in msgs:
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            h = (m.get("hash") or "").strip()
            if not role or not text:
                continue

            self.record_message(role, text, h)

            if role == "user":
                self.print_user(text)
            elif role == "assistant":
                # Avoid duplication: stream handles assistant most of the time
                if not self.stream_open:
                    print(f"\neidos> {text}\n", flush=True)

    def handle_assistant_stream(self, data: dict):
        mode = data.get("mode", "delta")
        delta = data.get("delta", "")
        full = data.get("full", "")

        self._open_stream()

        if mode == "reset":
            # The UI likely replaced the assistant DOM node; replace output line.
            sys.stdout.write("\n[stream reset]\n")
            sys.stdout.write("eidos> " + full)
            self.stream_buf = full
        else:
            sys.stdout.write(delta)
            self.stream_buf += delta

        self.stream_last_full = full or self.stream_last_full
        sys.stdout.flush()

    def handle_assistant_stable(self, data: dict):
        full = (data.get("full") or "").strip()
        if not full:
            return

        # Prevent double-finalize for the same stable text
        if self.stream_finalized_for == full:
            return
        self.stream_finalized_for = full

        # Ensure buffer matches the stable full
        self.stream_buf = full

        # Record + close stream
        self.record_message("assistant", full, h="")
        self._close_stream()


HELP_TEXT = """Commands:
  /new              start a new chat (best-effort)
  /reset            reload page
  /persist          save login/session state (storage_state)
  /save [path]      save transcript as markdown (default ./transcript.md)
  /quit             exit
  /help             show this help
"""


async def run_client(ws_url: str):
    state = ClientState()
    delay = RECONNECT_DELAY_S

    while True:
        try:
            async with websockets.connect(ws_url, max_size=16 * 1024 * 1024) as ws:
                state.print_status(f"Connected to {ws_url}")
                delay = RECONNECT_DELAY_S

                async def reader():
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            state.print_error("Bad JSON from server.")
                            continue

                        t = msg.get("type")
                        data = msg.get("data")

                        if t == "status":
                            state.print_status(str(data))
                        elif t == "error":
                            state.print_error(str(data))
                        elif t == "messages":
                            state.handle_messages(data or [])
                        elif t == "assistant_stream":
                            state.handle_assistant_stream(data or {})
                        elif t == "assistant_stable":
                            state.handle_assistant_stable(data or {})
                        else:
                            state.print_error(f"Unknown event type: {t!r}")

                async def writer():
                    loop = asyncio.get_running_loop()
                    while True:
                        text = await loop.run_in_executor(None, lambda: input("you> ").strip())
                        if not text:
                            continue

                        if text.startswith("/"):
                            parts = text.split(maxsplit=1)
                            cmd = parts[0].lower()
                            arg = parts[1] if len(parts) > 1 else ""

                            if cmd == "/help":
                                print(HELP_TEXT, flush=True)
                                continue

                            if cmd == "/quit":
                                await ws.send(json.dumps({"type": "quit", "data": ""}))
                                return

                            if cmd == "/new":
                                await ws.send(json.dumps({"type": "new", "data": ""}))
                                continue

                            if cmd == "/reset":
                                await ws.send(json.dumps({"type": "reset", "data": ""}))
                                continue

                            if cmd == "/persist":
                                await ws.send(json.dumps({"type": "persist", "data": ""}))
                                continue

                            if cmd == "/save":
                                path = arg.strip() or "./transcript.md"
                                try:
                                    save_markdown(state.entries, path)
                                    state.print_status(f"Saved transcript → {path}")
                                except Exception as e:
                                    state.print_error(f"Save failed: {e!r}")
                                continue

                            state.print_error("Unknown command. Type /help.")
                            continue

                        # normal message
                        await ws.send(json.dumps({"type": "send", "data": text}))

                await asyncio.gather(reader(), writer())
                return

        except KeyboardInterrupt:
            print("\nbye.\n", flush=True)
            return
        except Exception as e:
            print(f"[warn] connection failed: {e!r}", flush=True)
            print(f"[warn] reconnecting in {delay:.1f}s...", flush=True)
            await asyncio.sleep(delay)
            delay = min(MAX_RECONNECT_DELAY_S, delay * 1.5)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eidos sidecar client (terminal UI).")
    p.add_argument("--ws", default=DEFAULT_WS, help=f"WebSocket URL (default {DEFAULT_WS})")
    return p


def main():
    args = build_argparser().parse_args()
    print("Eidos sidecar client. Type /help for commands.\n", flush=True)
    asyncio.run(run_client(args.ws))


if __name__ == "__main__":
    main()
