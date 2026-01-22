from pathlib import Path
from typing import Any, Dict
import pytest
import srsly
from typer.testing import CliRunner
from weasel import app
from weasel.cli.document import MARKER_END, MARKER_IGNORE, MARKER_START, MARKER_TAGS
def test_raise_error_no_config():
    result = runner.invoke(app, ['document'])
    assert result.exit_code == 1