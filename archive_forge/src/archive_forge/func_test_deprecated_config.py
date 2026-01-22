from __future__ import annotations
import traceback
from contextlib import contextmanager
import pytest
import dask
from dask.utils import shorten_traceback
def test_deprecated_config(tmp_path):
    """Test config override in the format between 2023.6.1 and 2023.8.1"""
    d = {}
    dask.config.refresh(config=d)
    actual = dask.config.get('admin.traceback.shorten', config=d)
    assert isinstance(actual, list) and len(actual) > 2
    d = {}
    with open(tmp_path / 'dask.yaml', 'w') as fh:
        fh.write('\n            admin:\n              traceback:\n                shorten:\n                  when:\n                    - dask/base.py\n                  what:\n                    - dask/core.py\n            ')
    with pytest.warns(FutureWarning):
        dask.config.refresh(config=d, paths=[tmp_path])
    actual = dask.config.get('admin.traceback.shorten', config=d)
    assert actual == ['dask/core.py']