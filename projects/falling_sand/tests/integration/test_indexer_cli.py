import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_indexer_cli(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    source_root.mkdir(parents=True)
    tests_root.mkdir(parents=True)
    (source_root / "pkg.py").write_text("def demo():\n    return 1\n", encoding="utf-8")
    (tests_root / "test_pkg.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")

    output = tmp_path / "index.json"
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "falling_sand.indexer",
            "--source-root",
            str(source_root),
            "--tests-root",
            str(tests_root),
            "--output",
            str(output),
        ],
        check=True,
        env=env,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stats"]["total"] == 2
    assert payload["schema_version"] >= 2
