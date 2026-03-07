from __future__ import annotations

import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "python_test_matrix.py"


def _run(tmp_path: Path, content: str = "", force_all: bool = False) -> dict[str, list[dict[str, str]]]:
    changed = tmp_path / "changed.txt"
    changed.write_text(content, encoding="utf-8")
    cmd = [str(SCRIPT), "--changed-files", str(changed)]
    if force_all:
        cmd.append("--all")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_matrix_selects_changed_component(tmp_path: Path) -> None:
    payload = _run(tmp_path, "memory_forge/src/memory_forge/core/tiered_memory.py\n")
    components = [item["component"] for item in payload["include"]]
    assert components == ["memory_forge"]


def test_matrix_includes_all_on_global_trigger(tmp_path: Path) -> None:
    payload = _run(tmp_path, ".github/workflows/ci.yml\n")
    components = {item["component"] for item in payload["include"]}
    assert "agent_forge" in components
    assert "code_forge" in components
    assert "word_forge" in components


def test_matrix_falls_back_to_core_set_for_docs_only(tmp_path: Path) -> None:
    payload = _run(tmp_path, "docs/readme.md\n")
    components = [item["component"] for item in payload["include"]]
    assert components == [
        "agent_forge",
        "code_forge",
        "knowledge_forge",
        "memory_forge",
        "eidos_mcp",
        "scripts",
        "web_interface_forge",
    ]
