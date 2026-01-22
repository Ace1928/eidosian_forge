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
def test_core_file():
    assert 'temporary-directory' in dask.config.config
    assert 'dataframe' in dask.config.config
    assert 'compression' in dask.config.get('dataframe.shuffle')