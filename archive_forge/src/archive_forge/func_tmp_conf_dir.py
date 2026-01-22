from __future__ import annotations
import importlib
import json
import pathlib
import platform
import sys
import click
import pytest
import yaml
from click.testing import CliRunner
import dask
import dask.cli
from dask._compatibility import importlib_metadata
@pytest.fixture
def tmp_conf_dir(tmpdir, monkeypatch):
    monkeypatch.setenv('DASK_CONFIG', str(tmpdir))
    originals = dask.config.__dict__.copy()
    dask.config = importlib.reload(dask.config)
    dask.config.paths = [str(tmpdir)]
    try:
        yield pathlib.Path(tmpdir)
    finally:
        dask.config = importlib.reload(dask.config)
        dask.config.__dict__.update(originals)