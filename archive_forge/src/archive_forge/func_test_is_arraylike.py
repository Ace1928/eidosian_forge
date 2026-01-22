from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_is_arraylike():
    np = pytest.importorskip('numpy')
    assert is_arraylike(0) is False
    assert is_arraylike(()) is False
    assert is_arraylike(0) is False
    assert is_arraylike([]) is False
    assert is_arraylike([0]) is False
    assert is_arraylike(np.empty(())) is True
    assert is_arraylike(np.empty((0,))) is True
    assert is_arraylike(np.empty((0, 0))) is True