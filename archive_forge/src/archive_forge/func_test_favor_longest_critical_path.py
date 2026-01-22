from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_favor_longest_critical_path(abcde):
    """

       a
       |
    d  b  e
     \\ | /
       c

    """
    a, b, c, d, e = abcde
    dsk = {c: (f,), d: (f, c), e: (f, c), b: (f, c), a: (f, b)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert o[d] > o[b]
    assert o[e] > o[b]