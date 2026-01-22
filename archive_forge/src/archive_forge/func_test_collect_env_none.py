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
def test_collect_env_none(monkeypatch):
    monkeypatch.setenv('DASK_FOO', 'bar')
    config = collect([])
    assert config.get('foo') == 'bar'