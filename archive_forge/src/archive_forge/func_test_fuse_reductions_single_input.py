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
def test_fuse_reductions_single_input():

    def f(*args):
        return args
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a', 'a'), 'c': (f, 'b1', 'b2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    assert fuse(d, ave_width=2, rename_keys=False) == with_deps({'a': 1, 'c': (f, (f, 'a'), (f, 'a', 'a'))})
    assert fuse(d, ave_width=2, rename_keys=True) == with_deps({'a': 1, 'b1-b2-c': (f, (f, 'a'), (f, 'a', 'a')), 'c': 'b1-b2-c'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a', 'a'), 'b3': (f, 'a', 'a', 'a'), 'c': (f, 'b1', 'b2', 'b3')}
    assert fuse(d, ave_width=2.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=2.9, rename_keys=True) == with_deps(d)
    assert fuse(d, ave_width=3, rename_keys=False) == with_deps({'a': 1, 'c': (f, (f, 'a'), (f, 'a', 'a'), (f, 'a', 'a', 'a'))})
    assert fuse(d, ave_width=3, rename_keys=True) == with_deps({'a': 1, 'b1-b2-b3-c': (f, (f, 'a'), (f, 'a', 'a'), (f, 'a', 'a', 'a')), 'c': 'b1-b2-b3-c'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'c': (f, 'a', 'b1', 'b2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    assert fuse(d, ave_width=2, rename_keys=False) == with_deps({'a': 1, 'c': (f, 'a', (f, 'a'), (f, 'a'))})
    assert fuse(d, ave_width=2, rename_keys=True) == with_deps({'a': 1, 'b1-b2-c': (f, 'a', (f, 'a'), (f, 'a')), 'c': 'b1-b2-c'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'c': (f, 'b1', 'b2'), 'd1': (f, 'c'), 'd2': (f, 'c'), 'e': (f, 'd1', 'd2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    assert fuse(d, ave_width=2, rename_keys=False) == with_deps({'a': 1, 'c': (f, (f, 'a'), (f, 'a')), 'e': (f, (f, 'c'), (f, 'c'))})
    assert fuse(d, ave_width=2, rename_keys=True) == with_deps({'a': 1, 'b1-b2-c': (f, (f, 'a'), (f, 'a')), 'd1-d2-e': (f, (f, 'c'), (f, 'c')), 'c': 'b1-b2-c', 'e': 'd1-d2-e'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'b3': (f, 'a'), 'b4': (f, 'a'), 'c1': (f, 'b1', 'b2'), 'c2': (f, 'b3', 'b4'), 'd': (f, 'c1', 'c2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    expected = with_deps({'a': 1, 'c1': (f, (f, 'a'), (f, 'a')), 'c2': (f, (f, 'a'), (f, 'a')), 'd': (f, 'c1', 'c2')})
    assert fuse(d, ave_width=2, rename_keys=False) == expected
    assert fuse(d, ave_width=2.9, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-c1': (f, (f, 'a'), (f, 'a')), 'b3-b4-c2': (f, (f, 'a'), (f, 'a')), 'd': (f, 'c1', 'c2'), 'c1': 'b1-b2-c1', 'c2': 'b3-b4-c2'})
    assert fuse(d, ave_width=2, rename_keys=True) == expected
    assert fuse(d, ave_width=2.9, rename_keys=True) == expected
    assert fuse(d, ave_width=3, rename_keys=False) == with_deps({'a': 1, 'd': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))})
    assert fuse(d, ave_width=3, rename_keys=True) == with_deps({'a': 1, 'b1-b2-b3-b4-c1-c2-d': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'd': 'b1-b2-b3-b4-c1-c2-d'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'b3': (f, 'a'), 'b4': (f, 'a'), 'b5': (f, 'a'), 'b6': (f, 'a'), 'b7': (f, 'a'), 'b8': (f, 'a'), 'c1': (f, 'b1', 'b2'), 'c2': (f, 'b3', 'b4'), 'c3': (f, 'b5', 'b6'), 'c4': (f, 'b7', 'b8'), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'e': (f, 'd1', 'd2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    expected = with_deps({'a': 1, 'c1': (f, (f, 'a'), (f, 'a')), 'c2': (f, (f, 'a'), (f, 'a')), 'c3': (f, (f, 'a'), (f, 'a')), 'c4': (f, (f, 'a'), (f, 'a')), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'e': (f, 'd1', 'd2')})
    assert fuse(d, ave_width=2, rename_keys=False) == expected
    assert fuse(d, ave_width=2.9, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-c1': (f, (f, 'a'), (f, 'a')), 'b3-b4-c2': (f, (f, 'a'), (f, 'a')), 'b5-b6-c3': (f, (f, 'a'), (f, 'a')), 'b7-b8-c4': (f, (f, 'a'), (f, 'a')), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'e': (f, 'd1', 'd2'), 'c1': 'b1-b2-c1', 'c2': 'b3-b4-c2', 'c3': 'b5-b6-c3', 'c4': 'b7-b8-c4'})
    assert fuse(d, ave_width=2, rename_keys=True) == expected
    assert fuse(d, ave_width=2.9, rename_keys=True) == expected
    expected = with_deps({'a': 1, 'd1': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'd2': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'e': (f, 'd1', 'd2')})
    assert fuse(d, ave_width=3, rename_keys=False) == expected
    assert fuse(d, ave_width=4.6, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-b3-b4-c1-c2-d1': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'b5-b6-b7-b8-c3-c4-d2': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'e': (f, 'd1', 'd2'), 'd1': 'b1-b2-b3-b4-c1-c2-d1', 'd2': 'b5-b6-b7-b8-c3-c4-d2'})
    assert fuse(d, ave_width=3, rename_keys=True) == expected
    assert fuse(d, ave_width=4.6, rename_keys=True) == expected
    assert fuse(d, ave_width=4.7, rename_keys=False) == with_deps({'a': 1, 'e': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))))})
    assert fuse(d, ave_width=4.7, rename_keys=True) == with_deps({'a': 1, 'b1-b2-b3-b4-b5-b6-b7-b8-c1-c2-c3-c4-d1-d2-e': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), 'e': 'b1-b2-b3-b4-b5-b6-b7-b8-c1-c2-c3-c4-d1-d2-e'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'b3': (f, 'a'), 'b4': (f, 'a'), 'b5': (f, 'a'), 'b6': (f, 'a'), 'b7': (f, 'a'), 'b8': (f, 'a'), 'b9': (f, 'a'), 'b10': (f, 'a'), 'b11': (f, 'a'), 'b12': (f, 'a'), 'b13': (f, 'a'), 'b14': (f, 'a'), 'b15': (f, 'a'), 'b16': (f, 'a'), 'c1': (f, 'b1', 'b2'), 'c2': (f, 'b3', 'b4'), 'c3': (f, 'b5', 'b6'), 'c4': (f, 'b7', 'b8'), 'c5': (f, 'b9', 'b10'), 'c6': (f, 'b11', 'b12'), 'c7': (f, 'b13', 'b14'), 'c8': (f, 'b15', 'b16'), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'd3': (f, 'c5', 'c6'), 'd4': (f, 'c7', 'c8'), 'e1': (f, 'd1', 'd2'), 'e2': (f, 'd3', 'd4'), 'f': (f, 'e1', 'e2')}
    assert fuse(d, ave_width=1.9, rename_keys=False) == with_deps(d)
    assert fuse(d, ave_width=1.9, rename_keys=True) == with_deps(d)
    expected = with_deps({'a': 1, 'c1': (f, (f, 'a'), (f, 'a')), 'c2': (f, (f, 'a'), (f, 'a')), 'c3': (f, (f, 'a'), (f, 'a')), 'c4': (f, (f, 'a'), (f, 'a')), 'c5': (f, (f, 'a'), (f, 'a')), 'c6': (f, (f, 'a'), (f, 'a')), 'c7': (f, (f, 'a'), (f, 'a')), 'c8': (f, (f, 'a'), (f, 'a')), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'd3': (f, 'c5', 'c6'), 'd4': (f, 'c7', 'c8'), 'e1': (f, 'd1', 'd2'), 'e2': (f, 'd3', 'd4'), 'f': (f, 'e1', 'e2')})
    assert fuse(d, ave_width=2, rename_keys=False) == expected
    assert fuse(d, ave_width=2.9, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-c1': (f, (f, 'a'), (f, 'a')), 'b3-b4-c2': (f, (f, 'a'), (f, 'a')), 'b5-b6-c3': (f, (f, 'a'), (f, 'a')), 'b7-b8-c4': (f, (f, 'a'), (f, 'a')), 'b10-b9-c5': (f, (f, 'a'), (f, 'a')), 'b11-b12-c6': (f, (f, 'a'), (f, 'a')), 'b13-b14-c7': (f, (f, 'a'), (f, 'a')), 'b15-b16-c8': (f, (f, 'a'), (f, 'a')), 'd1': (f, 'c1', 'c2'), 'd2': (f, 'c3', 'c4'), 'd3': (f, 'c5', 'c6'), 'd4': (f, 'c7', 'c8'), 'e1': (f, 'd1', 'd2'), 'e2': (f, 'd3', 'd4'), 'f': (f, 'e1', 'e2'), 'c1': 'b1-b2-c1', 'c2': 'b3-b4-c2', 'c3': 'b5-b6-c3', 'c4': 'b7-b8-c4', 'c5': 'b10-b9-c5', 'c6': 'b11-b12-c6', 'c7': 'b13-b14-c7', 'c8': 'b15-b16-c8'})
    assert fuse(d, ave_width=2, rename_keys=True) == expected
    assert fuse(d, ave_width=2.9, rename_keys=True) == expected
    expected = with_deps({'a': 1, 'd1': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'd2': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'd3': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'd4': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'e1': (f, 'd1', 'd2'), 'e2': (f, 'd3', 'd4'), 'f': (f, 'e1', 'e2')})
    assert fuse(d, ave_width=3, rename_keys=False) == expected
    assert fuse(d, ave_width=4.6, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-b3-b4-c1-c2-d1': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'b5-b6-b7-b8-c3-c4-d2': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'b10-b11-b12-b9-c5-c6-d3': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'b13-b14-b15-b16-c7-c8-d4': (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), 'e1': (f, 'd1', 'd2'), 'e2': (f, 'd3', 'd4'), 'f': (f, 'e1', 'e2'), 'd1': 'b1-b2-b3-b4-c1-c2-d1', 'd2': 'b5-b6-b7-b8-c3-c4-d2', 'd3': 'b10-b11-b12-b9-c5-c6-d3', 'd4': 'b13-b14-b15-b16-c7-c8-d4'})
    assert fuse(d, ave_width=3, rename_keys=True) == expected
    assert fuse(d, ave_width=4.6, rename_keys=True) == expected
    expected = with_deps({'a': 1, 'e1': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), 'e2': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), 'f': (f, 'e1', 'e2')})
    assert fuse(d, ave_width=4.7, rename_keys=False) == expected
    assert fuse(d, ave_width=7.4, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b1-b2-b3-b4-b5-b6-b7-b8-c1-c2-c3-c4-d1-d2-e1': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), 'b10-b11-b12-b13-b14-b15-b16-b9-c5-c6-c7-c8-d3-d4-e2': (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), 'f': (f, 'e1', 'e2'), 'e1': 'b1-b2-b3-b4-b5-b6-b7-b8-c1-c2-c3-c4-d1-d2-e1', 'e2': 'b10-b11-b12-b13-b14-b15-b16-b9-c5-c6-c7-c8-d3-d4-e2'})
    assert fuse(d, ave_width=4.7, rename_keys=True) == expected
    assert fuse(d, ave_width=7.4, rename_keys=True) == expected
    assert fuse(d, ave_width=7.5, rename_keys=False) == with_deps({'a': 1, 'f': (f, (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))))})
    assert fuse(d, ave_width=7.5, rename_keys=True) == with_deps({'a': 1, 'b1-b10-b11-b12-b13-b14-b15-b16-b2-b3-b4-b5-b6-b7-b8-b9-c1-c2-c3-c4-c5-c6-c7-c8-d1-d2-d3-d4-e1-e2-f': (f, (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a')))), (f, (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))), (f, (f, (f, 'a'), (f, 'a')), (f, (f, 'a'), (f, 'a'))))), 'f': 'b1-b10-b11-b12-b13-b14-b15-b16-b2-b3-b4-b5-b6-b7-b8-b9-c1-c2-c3-c4-c5-c6-c7-c8-d1-d2-d3-d4-e1-e2-f'})
    d = {'a': 1, 'b': (f, 'a')}
    assert fuse(d, ave_width=1, rename_keys=False) == with_deps({'b': (f, 1)})
    assert fuse(d, ave_width=1, rename_keys=True) == with_deps({'a-b': (f, 1), 'b': 'a-b'})
    d = {'a': 1, 'b': (f, 'a'), 'c': (f, 'b'), 'd': (f, 'c')}
    assert fuse(d, ave_width=1, rename_keys=False) == with_deps({'d': (f, (f, (f, 1)))})
    assert fuse(d, ave_width=1, rename_keys=True) == with_deps({'a-b-c-d': (f, (f, (f, 1))), 'd': 'a-b-c-d'})
    d = {'a': 1, 'b': (f, 'a'), 'c': (f, 'a', 'b'), 'd': (f, 'a', 'c')}
    assert fuse(d, ave_width=1, rename_keys=False) == with_deps({'a': 1, 'd': (f, 'a', (f, 'a', (f, 'a')))})
    assert fuse(d, ave_width=1, rename_keys=True) == with_deps({'a': 1, 'b-c-d': (f, 'a', (f, 'a', (f, 'a'))), 'd': 'b-c-d'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'c1': (f, 'b1'), 'd1': (f, 'c1'), 'e1': (f, 'd1'), 'f': (f, 'e1', 'b2')}
    expected = with_deps({'a': 1, 'b2': (f, 'a'), 'e1': (f, (f, (f, (f, 'a')))), 'f': (f, 'e1', 'b2')})
    assert fuse(d, ave_width=1, rename_keys=False) == expected
    assert fuse(d, ave_width=1.9, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b2': (f, 'a'), 'b1-c1-d1-e1': (f, (f, (f, (f, 'a')))), 'f': (f, 'e1', 'b2'), 'e1': 'b1-c1-d1-e1'})
    assert fuse(d, ave_width=1, rename_keys=True) == expected
    assert fuse(d, ave_width=1.9, rename_keys=True) == expected
    assert fuse(d, ave_width=2, rename_keys=False) == with_deps({'a': 1, 'f': (f, (f, (f, (f, (f, 'a')))), (f, 'a'))})
    assert fuse(d, ave_width=2, rename_keys=True) == with_deps({'a': 1, 'b1-b2-c1-d1-e1-f': (f, (f, (f, (f, (f, 'a')))), (f, 'a')), 'f': 'b1-b2-c1-d1-e1-f'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'c1': (f, 'a', 'b1'), 'd1': (f, 'a', 'c1'), 'e1': (f, 'a', 'd1'), 'f': (f, 'a', 'e1', 'b2')}
    expected = with_deps({'a': 1, 'b2': (f, 'a'), 'e1': (f, 'a', (f, 'a', (f, 'a', (f, 'a')))), 'f': (f, 'a', 'e1', 'b2')})
    assert fuse(d, ave_width=1, rename_keys=False) == expected
    assert fuse(d, ave_width=1.9, rename_keys=False) == expected
    expected = with_deps({'a': 1, 'b2': (f, 'a'), 'b1-c1-d1-e1': (f, 'a', (f, 'a', (f, 'a', (f, 'a')))), 'f': (f, 'a', 'e1', 'b2'), 'e1': 'b1-c1-d1-e1'})
    assert fuse(d, ave_width=1, rename_keys=True) == expected
    assert fuse(d, ave_width=1.9, rename_keys=True) == expected
    assert fuse(d, ave_width=2, rename_keys=False) == with_deps({'a': 1, 'f': (f, 'a', (f, 'a', (f, 'a', (f, 'a', (f, 'a')))), (f, 'a'))})
    assert fuse(d, ave_width=2, rename_keys=True) == with_deps({'a': 1, 'b1-b2-c1-d1-e1-f': (f, 'a', (f, 'a', (f, 'a', (f, 'a', (f, 'a')))), (f, 'a')), 'f': 'b1-b2-c1-d1-e1-f'})
    d = {'a': 1, 'b1': (f, 'a'), 'b2': (f, 'a'), 'b3': (f, 'a'), 'c1': (f, 'b1'), 'c2': (f, 'b2'), 'c3': (f, 'b3'), 'd1': (f, 'c1'), 'd2': (f, 'c2'), 'd3': (f, 'c3'), 'e': (f, 'd1', 'd2', 'd3'), 'f': (f, 'e'), 'g': (f, 'f')}
    assert fuse(d, ave_width=1, rename_keys=False) == with_deps({'a': 1, 'd1': (f, (f, (f, 'a'))), 'd2': (f, (f, (f, 'a'))), 'd3': (f, (f, (f, 'a'))), 'g': (f, (f, (f, 'd1', 'd2', 'd3')))})
    assert fuse(d, ave_width=1, rename_keys=True) == with_deps({'a': 1, 'b1-c1-d1': (f, (f, (f, 'a'))), 'b2-c2-d2': (f, (f, (f, 'a'))), 'b3-c3-d3': (f, (f, (f, 'a'))), 'e-f-g': (f, (f, (f, 'd1', 'd2', 'd3'))), 'd1': 'b1-c1-d1', 'd2': 'b2-c2-d2', 'd3': 'b3-c3-d3', 'g': 'e-f-g'})
    d = {'a': 1, 'b': (f, 'a'), 'c': (f, 'b'), 'd': (f, 'b', 'c'), 'e': (f, 'd'), 'f': (f, 'e'), 'g': (f, 'd', 'f')}
    assert fuse(d, ave_width=1, rename_keys=False) == with_deps({'b': (f, 1), 'd': (f, 'b', (f, 'b')), 'g': (f, 'd', (f, (f, 'd')))})
    assert fuse(d, ave_width=1, rename_keys=True) == with_deps({'a-b': (f, 1), 'c-d': (f, 'b', (f, 'b')), 'e-f-g': (f, 'd', (f, (f, 'd'))), 'b': 'a-b', 'd': 'c-d', 'g': 'e-f-g'})