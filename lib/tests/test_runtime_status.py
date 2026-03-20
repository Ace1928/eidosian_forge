from __future__ import annotations

import json
from pathlib import Path

from eidosian_runtime.runtime_status import write_runtime_status


def test_write_runtime_status_writes_status_and_history(tmp_path: Path) -> None:
    status_path = tmp_path / "runtime" / "component" / "status.json"
    history_path = tmp_path / "runtime" / "component" / "history.jsonl"

    payload = write_runtime_status(
        status_path,
        contract="eidos.test.status.v1",
        component="test_component",
        status="running",
        phase="phase_a",
        message="hello",
        history_path=history_path,
        session_id="session:test",
    )

    assert payload["contract"] == "eidos.test.status.v1"
    assert payload["component"] == "test_component"
    assert payload["status"] == "running"
    assert payload["phase"] == "phase_a"
    assert payload["session_id"] == "session:test"

    written = json.loads(status_path.read_text(encoding="utf-8"))
    assert written["message"] == "hello"
    history_rows = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    assert json.loads(history_rows[0])["status"] == "running"
