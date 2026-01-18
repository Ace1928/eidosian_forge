"""Tests for the glossary generation tool."""

from pathlib import Path
import subprocess
import sys


def test_generate_glossary_cli(tmp_path: Path) -> None:
    """Run the CLI and validate that expected symbols appear."""

    glossary_file = Path("knowledge/glossary_reference.md")
    backup = glossary_file.read_text() if glossary_file.exists() else ""
    try:
        result = subprocess.run(
            [sys.executable, "tools/generate_glossary.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = glossary_file.read_text()
        assert "EidosCore" in content
        assert "UtilityAgent" in content
    finally:
        glossary_file.write_text(backup)
