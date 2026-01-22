from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_avoid_broker_nodes(abcde):
    """

    b0    b1  b2
    |      \\  /
    a0      a1

    There are good arguments for both a0 or a1 to run first. Regardless of what
    we choose to run first, we should finish the computation branch before
    moving to the other one
    """
    a, b, c, d, e = abcde
    dsk = {(a, 0): (f,), (a, 1): (f,), (b, 0): (f, (a, 0)), (b, 1): (f, (a, 1)), (b, 2): (f, (a, 1))}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[a, 1] < o[b, 0] or (o[b, 1] < o[a, 0] and o[b, 2] < o[a, 0])