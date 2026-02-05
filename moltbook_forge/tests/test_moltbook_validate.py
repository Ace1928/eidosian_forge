from __future__ import annotations

import json
from pathlib import Path

from moltbook_forge.moltbook_validate import main


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_validate_ok(tmp_path: Path) -> None:
    payload = {
        "raw_sha256": "x",
        "normalized_sha256": "y",
        "length_raw": 3,
        "length_normalized": 3,
        "line_count": 1,
        "word_count": 1,
        "non_ascii_ratio": 0.0,
        "truncated": False,
        "flags": [],
        "risk_score": 0.0,
        "text": "hey",
    }
    path = _write_payload(tmp_path, payload)
    out = tmp_path / "result.json"
    assert main(["--input", str(path), "--output", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["ok"] is True


def test_validate_errors(tmp_path: Path) -> None:
    payload = {"text": "missing fields"}
    path = _write_payload(tmp_path, payload)
    out = tmp_path / "result.json"
    assert main(["--input", str(path), "--output", str(out)]) == 1
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["ok"] is False
