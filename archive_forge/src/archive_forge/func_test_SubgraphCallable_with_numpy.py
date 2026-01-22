from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def test_SubgraphCallable_with_numpy():
    np = pytest.importorskip('numpy')
    dsk1 = {'a': np.arange(10)}
    f1 = SubgraphCallable(dsk1, 'a', [None], name='test')
    f2 = SubgraphCallable(dsk1, 'a', [None], name='test')
    assert f1 == f2
    dsk2 = {'a': np.arange(10) + 1}
    f3 = SubgraphCallable(dsk2, 'a', [None], name='test')
    assert f1 == f3
    f4 = SubgraphCallable(dsk1, 'a', [None], name='test2')
    assert f1 != f4