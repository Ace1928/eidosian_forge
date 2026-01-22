from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_local_parents_of_reduction(abcde):
    """

            c1
            |
        b1  c2
        |  /|
    a1  b2  c3
    |  /|
    a2  b3
    |
    a3

    Prefer to finish a1 stack before proceeding to b2
    """
    a, b, c, d, e = abcde
    a1, a2, a3 = (a + i for i in '123')
    b1, b2, b3 = (b + i for i in '123')
    c1, c2, c3 = (c + i for i in '123')
    expected = [a3, a2, a1, b3, b2, b1, c3, c2, c1]
    log = []

    def f(x):

        def _(*args):
            log.append(x)
        return _
    dsk = {a3: (f(a3),), a2: (f(a2), a3), a1: (f(a1), a2), b3: (f(b3),), b2: (f(b2), b3, a2), b1: (f(b1), b2), c3: (f(c3),), c2: (f(c2), c3, b2), c1: (f(c1), c2)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    dask.get(dsk, [a1, b1, c1])
    assert log == expected