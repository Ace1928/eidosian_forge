from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_falling_sand_profile_index_output(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    _write_file(source_root / "pkg.py", "def demo():\n    return 1\n")
    _write_file(tests_root / "test_pkg.py", "def test_demo():\n    assert True\n")

    output = tmp_path / "profile.pstats"
    script_path = Path(__file__).resolve().parents[1] / "src" / "falling_sand" / "scripts" / "profile_index.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--source-root",
            str(source_root),
            "--tests-root",
            str(tests_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert output.exists()
