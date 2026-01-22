from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_iskey_numpy_types():
    np = pytest.importorskip('numpy')
    one = np.int64(1)
    assert not iskey(one)
    assert not iskey(('foo', one))