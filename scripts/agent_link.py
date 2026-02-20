#!/usr/bin/env python3
"""Eidos cross-machine agent link over synced filesystem."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BRIDGE_DEFAULT = Path.home() / ".eidosian" / "agent_bridge"


@dataclass
class BridgePaths:
    root: Path
    messages: Path
    acks: Path
    presence: Path
    state: Path
    logs: Path
    context: Path
    bin: Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def ensure_dirs(root: Path) -> BridgePaths:
    root = root.expanduser().resolve()
    paths = BridgePaths(
        root=root,
        messages=root / "messages",
        acks=root / "acks",
        presence=root / "presence",
        state=root / "state",
        logs=root / "logs",
        context=root / "context",
        bin=root / "bin",
    )
    for path in [
        paths.root,
        paths.messages,
        paths.acks,
        paths.presence,
        paths.state,
        paths.logs,
        paths.context,
        paths.bin,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def default_agent_id() -> str:
    return socket.gethostname()


def message_file_name(msg_id: str) -> str:
    return f"{msg_id}.json"


def write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def build_message_id(sender: str, target: str) -> str:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}__{sender}__{target}__{uuid.uuid4().hex[:12]}"


def send_message(
    bridge: BridgePaths,
    sender: str,
    target: str,
    topic: str,
    body: str,
    priority: str,
    requires_ack: bool,
    tags: list[str],
) -> Path:
    msg_id = build_message_id(sender=sender, target=target)
    payload = {
        "id": msg_id,
        "timestamp_utc": utc_iso(),
        "from": sender,
        "to": target,
        "topic": topic,
        "priority": priority,
        "requires_ack": requires_ack,
        "tags": tags,
        "body": body,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    out = bridge.messages / message_file_name(msg_id)
    write_atomic(out, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    return out


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ack_path(bridge: BridgePaths, msg_id: str, agent_id: str) -> Path:
    return bridge.acks / f"{msg_id}__{agent_id}.ack"


def write_ack(bridge: BridgePaths, msg_id: str, agent_id: str) -> Path:
    out = ack_path(bridge=bridge, msg_id=msg_id, agent_id=agent_id)
    if not out.exists():
        out.write_text(f"acked_utc={utc_iso()}\n", encoding="utf-8")
    return out


def is_targeted(payload: dict[str, Any], agent_id: str) -> bool:
    target = str(payload.get("to", "all"))
    if target == "all":
        return True
    if target == agent_id:
        return True
    if target == socket.gethostname():
        return True
    return False


def cursor_file(bridge: BridgePaths, agent_id: str) -> Path:
    return bridge.state / f"{agent_id}.cursor"


def load_cursor(bridge: BridgePaths, agent_id: str) -> str:
    path = cursor_file(bridge=bridge, agent_id=agent_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def save_cursor(bridge: BridgePaths, agent_id: str, value: str) -> None:
    if not value:
        return
    write_atomic(cursor_file(bridge=bridge, agent_id=agent_id), value + "\n")


def update_presence(bridge: BridgePaths, agent_id: str, status: str = "online") -> None:
    payload = {
        "agent_id": agent_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "status": status,
        "heartbeat_utc": utc_iso(),
    }
    out = bridge.presence / f"{agent_id}.json"
    write_atomic(out, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def read_messages_since(bridge: BridgePaths, last_file: str) -> list[Path]:
    files = sorted(bridge.messages.glob("*.json"))
    if not last_file:
        return files
    return [path for path in files if path.name > last_file]


def print_message(payload: dict[str, Any], already_acked: bool) -> None:
    ts = payload.get("timestamp_utc", "")
    sender = payload.get("from", "?")
    target = payload.get("to", "?")
    topic = payload.get("topic", "")
    priority = payload.get("priority", "normal")
    msg_id = payload.get("id", "")
    print(f"[{ts}] {sender} -> {target} topic={topic} priority={priority}")
    body = str(payload.get("body", "")).strip()
    if body:
        print(body)
    print(f"id={msg_id} acked={already_acked}")
    print("---")


def command_send(args: argparse.Namespace) -> int:
    bridge = ensure_dirs(Path(args.bridge))
    sender = args.from_agent or default_agent_id()
    body = args.body
    if args.body_file:
        body = Path(args.body_file).expanduser().read_text(encoding="utf-8")
    out = send_message(
        bridge=bridge,
        sender=sender,
        target=args.to,
        topic=args.topic,
        body=body,
        priority=args.priority,
        requires_ack=args.requires_ack,
        tags=args.tag,
    )
    print(out)
    return 0


def command_watch(args: argparse.Namespace) -> int:
    bridge = ensure_dirs(Path(args.bridge))
    agent_id = args.agent_id or default_agent_id()
    poll = max(0.2, float(args.poll_interval))
    heartbeat_interval = max(1.0, float(args.heartbeat_interval))
    last_hb = 0.0
    last_seen = load_cursor(bridge=bridge, agent_id=agent_id)

    while True:
        now = time.time()
        if now - last_hb >= heartbeat_interval:
            update_presence(bridge=bridge, agent_id=agent_id, status="online")
            last_hb = now

        messages = read_messages_since(bridge=bridge, last_file=last_seen)
        for msg_path in messages:
            payload = read_json(msg_path)
            last_seen = msg_path.name
            if payload is None:
                continue
            if not is_targeted(payload=payload, agent_id=agent_id):
                continue

            msg_id = str(payload.get("id", msg_path.stem))
            acked = ack_path(bridge=bridge, msg_id=msg_id, agent_id=agent_id).exists()
            if args.only_unacked and acked:
                continue

            print_message(payload=payload, already_acked=acked)
            if args.ack and not acked:
                write_ack(bridge=bridge, msg_id=msg_id, agent_id=agent_id)

        save_cursor(bridge=bridge, agent_id=agent_id, value=last_seen)
        time.sleep(poll)


def command_status(args: argparse.Namespace) -> int:
    bridge = ensure_dirs(Path(args.bridge))
    stale_seconds = max(1, int(args.stale_seconds))
    agent_id = args.agent_id or default_agent_id()
    now = utc_now()

    print(f"Bridge: {bridge.root}")
    print(f"Agent: {agent_id}")
    print("")

    print("Presence:")
    for presence_file in sorted(bridge.presence.glob("*.json")):
        payload = read_json(presence_file) or {}
        hb_text = str(payload.get("heartbeat_utc", ""))
        age = "unknown"
        status = str(payload.get("status", "unknown"))
        if hb_text:
            try:
                hb = datetime.fromisoformat(hb_text)
                if hb.tzinfo is None:
                    hb = hb.replace(tzinfo=timezone.utc)
                delta = int((now - hb).total_seconds())
                age = f"{delta}s"
                if delta > stale_seconds:
                    status = f"stale({status})"
            except Exception:
                pass
        print(f"  {presence_file.stem:24s} status={status:12s} age={age}")

    print("")
    print("Queue:")
    total = 0
    targeted = 0
    unacked = 0
    for msg_file in bridge.messages.glob("*.json"):
        payload = read_json(msg_file)
        if payload is None:
            continue
        total += 1
        if is_targeted(payload=payload, agent_id=agent_id):
            targeted += 1
            msg_id = str(payload.get("id", msg_file.stem))
            needs_ack = bool(payload.get("requires_ack", True))
            if needs_ack and not ack_path(bridge=bridge, msg_id=msg_id, agent_id=agent_id).exists():
                unacked += 1
    print(f"  total_messages={total}")
    print(f"  targeted_to_agent={targeted}")
    print(f"  unacked_for_agent={unacked}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", default=str(BRIDGE_DEFAULT), help="Bridge root directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_send = sub.add_parser("send", help="Send a message.")
    p_send.add_argument("--from-agent", default="", help="Sender id (default: hostname).")
    p_send.add_argument("--to", default="all", help="Target agent/host or 'all'.")
    p_send.add_argument("--topic", required=True, help="Topic string.")
    p_send.add_argument("--body", default="", help="Body text.")
    p_send.add_argument("--body-file", default="", help="Read body from file.")
    p_send.add_argument("--priority", default="normal", choices=["low", "normal", "high", "urgent"])
    p_send.add_argument("--requires-ack", action="store_true", default=True, help="Require ack marker.")
    p_send.add_argument("--tag", action="append", default=[], help="Optional tags.")
    p_send.set_defaults(func=command_send)

    p_watch = sub.add_parser("watch", help="Watch and print incoming messages continuously.")
    p_watch.add_argument("--agent-id", default="", help="Agent id (default: hostname).")
    p_watch.add_argument("--poll-interval", type=float, default=1.5, help="Polling interval seconds.")
    p_watch.add_argument("--heartbeat-interval", type=float, default=5.0, help="Presence heartbeat interval seconds.")
    p_watch.add_argument("--ack", action="store_true", help="Auto-ack displayed messages.")
    p_watch.add_argument("--only-unacked", action="store_true", help="Skip messages already acked by this agent.")
    p_watch.set_defaults(func=command_watch)

    p_status = sub.add_parser("status", help="Show bridge health and queue state.")
    p_status.add_argument("--agent-id", default="", help="Agent id (default: hostname).")
    p_status.add_argument("--stale-seconds", type=int, default=20, help="Presence staleness threshold.")
    p_status.set_defaults(func=command_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
