from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_connecting_to_roots_asym():
    dsk = {'a0': (f, 1), 'a1': (f, 1), 'a2': (f, 1), 'a3': (f, 1), 'a4': (f, 1), 'b1': (f, 'a0', 'a1'), 'c1': (f, 'b1'), 'c2': (f, 'b1'), 'd1': (f, 'c1', 'c2'), 'b2': (f, 'a2'), 'b3': (f, 'a3'), 'c3': (f, 'b3', 'b2'), 'd2': (f, 'b3', 'c3', 'a4')}
    dependencies, dependents = get_deps(dsk)
    connected_roots, max_dependents = _connecting_to_roots(dependencies, dependents)
    assert connected_roots == {'a0': set(), 'a1': set(), 'a2': set(), 'a3': set(), 'a4': set(), 'b1': {'a0', 'a1'}, 'c1': {'a0', 'a1'}, 'c2': {'a0', 'a1'}, 'd1': {'a0', 'a1'}, 'b2': {'a2'}, 'b3': {'a3'}, 'c3': {'a2', 'a3'}, 'd2': {'a2', 'a3', 'a4'}}
    assert max_dependents == {k: max((len(dependents[r]) for r in v), default=len(dependents[k])) for k, v in connected_roots.items()}