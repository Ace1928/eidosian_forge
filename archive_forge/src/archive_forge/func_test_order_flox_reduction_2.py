from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_order_flox_reduction_2(abcde):
    a, b, c, d, e = abcde
    dsk = {(a, 0): 0, (a, 1): 0, (a, 2): 0, (b, 0, 0, 0): (f, (a, 0)), (b, 0, 0, 1): (f, (a, 1)), (b, 0, 0, 2): (f, (a, 2)), (b, 0, 1, 0): (f, (a, 0)), (b, 0, 1, 1): (f, (a, 1)), (b, 0, 1, 2): (f, (a, 2)), (b, 1, 0, 0): (f, (a, 0)), (b, 1, 0, 1): (f, (a, 1)), (b, 1, 0, 2): (f, (a, 2)), (b, 1, 1, 0): (f, (a, 0)), (b, 1, 1, 1): (f, (a, 1)), (b, 1, 1, 2): (f, (a, 2)), (c, 0, 0): (f, [(b, 0, 0, 0), (b, 0, 0, 1), (b, 0, 0, 2)]), (c, 0, 1): (f, [(b, 0, 1, 0), (b, 0, 1, 1), (b, 0, 1, 2)]), (c, 1, 0): (f, [(b, 1, 0, 0), (b, 1, 0, 1), (b, 1, 0, 2)]), (c, 1, 1): (f, [(b, 1, 1, 0), (b, 1, 1, 1), (b, 1, 1, 2)]), (d, 0, 0): (c, 0, 0), (d, 0, 1): (c, 0, 1), (d, 1, 0): (c, 1, 0), (d, 1, 1): (c, 1, 1)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    final_nodes = sorted([(d, ix, jx) for ix in range(2) for jx in range(2)], key=o.__getitem__)
    for ix in range(1, len(final_nodes)):
        assert o[final_nodes[ix]] - o[final_nodes[ix - 1]] == 5