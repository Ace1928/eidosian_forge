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
def test__get_paths(monkeypatch):
    monkeypatch.delenv('DASK_CONFIG', raising=False)
    monkeypatch.delenv('DASK_ROOT_CONFIG', raising=False)
    monkeypatch.setattr(site, 'PREFIXES', [])
    expected = ['/etc/dask', os.path.join(sys.prefix, 'etc', 'dask'), os.path.join(os.path.expanduser('~'), '.config', 'dask')]
    paths = _get_paths()
    assert paths == expected
    assert len(paths) == len(set(paths))
    with monkeypatch.context() as m:
        m.setenv('DASK_CONFIG', 'foo-bar')
        paths = _get_paths()
        assert paths == expected + ['foo-bar']
        assert len(paths) == len(set(paths))
    with monkeypatch.context() as m:
        m.setenv('DASK_ROOT_CONFIG', 'foo-bar')
        paths = _get_paths()
        assert paths == ['foo-bar'] + expected[1:]
        assert len(paths) == len(set(paths))
    with monkeypatch.context() as m:
        prefix = os.path.join('include', 'this', 'path')
        m.setattr(site, 'PREFIXES', site.PREFIXES + [prefix])
        paths = _get_paths()
        assert os.path.join(prefix, 'etc', 'dask') in paths
        assert len(paths) == len(set(paths))