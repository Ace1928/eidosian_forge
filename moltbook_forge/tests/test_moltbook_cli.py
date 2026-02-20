from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from moltbook_forge.cli import main


def _write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "post.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_list() -> None:
    assert main(["--list"]) == 0


def test_cli_module_list() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "moltbook_forge", "--list"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert "available commands" in result.stdout.lower()


def test_cli_sanitize(tmp_path: Path) -> None:
    path = _write_text(tmp_path, "Hello Moltbook agent.")
    out = tmp_path / "sanitized.json"
    assert main(["sanitize", "--input", str(path), "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["length_normalized"] > 0
