from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_prefer_short_narrow(abcde):
    a, b, c, _, _ = abcde
    dsk = {(a, 0): 0, (b, 0): 0, (c, 0): 0, (c, 1): (f, (c, 0), (a, 0), (b, 0)), (a, 1): 1, (b, 1): 1, (c, 2): (f, (c, 1), (a, 1), (b, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[b, 0] < o[b, 1]
    assert o[b, 0] < o[c, 2]
    assert o[c, 1] < o[c, 2]