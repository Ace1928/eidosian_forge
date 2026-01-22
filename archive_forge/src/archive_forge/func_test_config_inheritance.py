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
def test_config_inheritance():
    config = collect_env({'DASK_INTERNAL_INHERIT_CONFIG': serialize({'array': {'svg': {'size': 150}}})})
    assert dask.config.get('array.svg.size', config=config) == 150