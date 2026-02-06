from __future__ import annotations

from pathlib import Path

from memory_forge.context.rescue import generate_rescue_kit


def test_generate_rescue_kit(tmp_path: Path) -> None:
    created = generate_rescue_kit(tmp_path)
    assert created
    index_path = tmp_path / "memory" / "index.json"
    last_session = tmp_path / "memory" / "last_session.md"
    assert index_path.exists()
    assert last_session.exists()
    # Idempotent second run
    created_second = generate_rescue_kit(tmp_path)
    assert created_second == []
