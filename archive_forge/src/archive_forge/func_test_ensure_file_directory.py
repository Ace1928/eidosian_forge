from __future__ import annotations
import os
import pathlib
import site
import stat
import sys
from collections import OrderedDict
from contextlib import contextmanager
import pytest
import yaml
import dask.config
from dask.config import (
@pytest.mark.parametrize('mkdir', [True, False])
def test_ensure_file_directory(mkdir, tmp_path):
    a = {'x': 1, 'y': {'a': 1}}
    source = tmp_path / 'source.yaml'
    dest = tmp_path / 'dest'
    source.write_text(yaml.dump(a))
    if mkdir:
        dest.mkdir()
    ensure_file(source=source, destination=dest)
    assert dest.is_dir()
    assert (dest / 'source.yaml').exists()