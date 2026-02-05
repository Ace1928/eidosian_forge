from __future__ import annotations

import json
from pathlib import Path

from moltbook_forge.cli import main


def _write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "post.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_list() -> None:
    assert main(["--list"]) == 0


def test_cli_sanitize(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "Hello Moltbook agent.")
    out = tmp_path / "sanitized.json"
    assert main(["sanitize", "--input", str(path), "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["length_normalized"] > 0
