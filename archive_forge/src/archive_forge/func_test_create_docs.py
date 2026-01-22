from pathlib import Path
from typing import Any, Dict
import pytest
import srsly
from typer.testing import CliRunner
from weasel import app
from weasel.cli.document import MARKER_END, MARKER_IGNORE, MARKER_START, MARKER_TAGS
def test_create_docs(project_yaml_file: Path):
    result = runner.invoke(app, ['document', str(project_yaml_file.parent)])
    conf_data = srsly.read_yaml(project_yaml_file)
    assert result.exit_code == 0
    assert conf_data['title'] in result.stdout