import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_project_assets(project_dir):
    asset_dir = project_dir / 'assets'
    assert not asset_dir.exists(), 'Assets dir is already present'
    result = CliRunner().invoke(app, ['project', 'assets', str(project_dir)])
    assert result.exit_code == 0
    assert (asset_dir / 'spacy-readme.md').is_file(), 'Assets not downloaded'
    result = CliRunner().invoke(app, ['project', 'assets', '--extra', str(project_dir)])
    assert result.exit_code == 0
    assert (asset_dir / 'citation.cff').is_file(), 'Extras not downloaded'