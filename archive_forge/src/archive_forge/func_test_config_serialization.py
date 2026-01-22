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
def test_config_serialization():
    with dask.config.set({'array.svg.size': dask.config.get('array.svg.size')}):
        serialized = serialize({'array': {'svg': {'size': 150}}})
        config = deserialize(serialized)
        dask.config.update(dask.config.global_config, config)
        assert dask.config.get('array.svg.size') == 150