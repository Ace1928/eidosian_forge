import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_convert_auto():
    with make_tempdir() as d_in, make_tempdir() as d_out:
        for f in ['data1.iob', 'data2.iob', 'data3.iob']:
            Path(d_in / f).touch()
        result = CliRunner().invoke(app, ['convert', str(d_in), str(d_out)])
        assert 'Generated output file' in result.stdout
        out_files = os.listdir(d_out)
        assert len(out_files) == 3
        assert 'data1.spacy' in out_files
        assert 'data2.spacy' in out_files
        assert 'data3.spacy' in out_files