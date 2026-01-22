from pathlib import Path
from typing import Any, Dict
import pytest
import srsly
from typer.testing import CliRunner
from weasel import app
from weasel.cli.document import MARKER_END, MARKER_IGNORE, MARKER_START, MARKER_TAGS
@pytest.fixture(scope='function')
def project_yaml_file(tmp_path_factory: pytest.TempPathFactory):
    test_dir = tmp_path_factory.mktemp('project')
    path = test_dir / 'project.yml'
    path.write_text(srsly.yaml_dumps(SAMPLE_PROJECT))
    return path