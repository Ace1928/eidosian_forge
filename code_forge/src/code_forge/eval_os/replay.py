from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_replay_key(
    *,
    task_id: str,
    config_id: str,
    command: str,
    workdir: str,
    timeout_sec: int,
    env_toggles: dict[str, Any],
) -> str:
    payload = {
        "task_id": task_id,
        "config_id": config_id,
        "command": command,
        "workdir": workdir,
        "timeout_sec": int(timeout_sec),
        "env_toggles": env_toggles,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ReplayStore:
    """Content-addressed record/replay storage for deterministic eval reruns."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, key: str, payload: dict[str, Any]) -> Path:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
