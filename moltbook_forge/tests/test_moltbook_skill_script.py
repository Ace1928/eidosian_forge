from __future__ import annotations

import subprocess
from pathlib import Path


def test_moltbook_skill_script_help() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "skills" / "moltbook" / "scripts" / "moltbook.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
