from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_dont_run_all_dependents_too_early(abcde):
    """From https://github.com/dask/dask-ml/issues/206#issuecomment-395873372"""
    a, b, c, d, e = abcde
    depth = 10
    dsk = {(a, 0): (f, 0), (b, 0): (f, 1), (c, 0): (f, 2), (d, 0): (f, (a, 0), (b, 0), (c, 0))}
    for i in range(1, depth):
        dsk[b, i] = (f, (b, 0))
        dsk[c, i] = (f, (c, 0))
        dsk[d, i] = (f, (d, i - 1), (b, i), (c, i))
    o = order(dsk)
    assert_topological_sort(dsk, o)
    expected = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    actual = sorted((v for (letter, num), v in o.items() if letter == d))
    assert expected == actual