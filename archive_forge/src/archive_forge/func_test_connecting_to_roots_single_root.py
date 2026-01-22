from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_connecting_to_roots_single_root():
    dsk = {'a': (f, 1), 'b1': (f, 'a'), 'b2': (f, 'a'), 'c': (f, 'b1', 'b2')}
    dependencies, dependents = get_deps(dsk)
    connected_roots, max_dependents = _connecting_to_roots(dependencies, dependents)
    assert connected_roots == {k: {'a'} if k != 'a' else set() for k in dsk}
    assert len({id(v) for v in connected_roots.values()}) == 2
    assert max_dependents == {'a': 2, 'b1': 2, 'b2': 2, 'c': 2}, max_dependents
    connected_roots, max_dependents = _connecting_to_roots(dependents, dependencies)
    assert connected_roots == {k: {'c'} if k != 'c' else set() for k in dsk}
    assert len({id(v) for v in connected_roots.values()}) == 2
    assert max_dependents == {'a': 2, 'b1': 2, 'b2': 2, 'c': 2}, max_dependents