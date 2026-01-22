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
def test_SubgraphCallable_eq():
    dsk1 = {'a': 1, 'b': 2, 'c': (add, 'd', 'e')}
    dsk2 = {'a': (inc, 0), 'b': (inc, 'a'), 'c': (add, 'd', 'e')}
    f1 = SubgraphCallable(dsk1, 'c', ['d', 'e'])
    f2 = SubgraphCallable(dsk2, 'c', ['d', 'e'])
    assert f1 != f2
    assert hash(f1) != hash(f2)
    assert tokenize(f1) != tokenize(f2)
    dsk1b = {'a': 1, 'b': 2, 'c': (add, 'd', 'e')}
    f1b = SubgraphCallable(dsk1b, 'c', ['d', 'e'])
    assert f1b.name == f1.name
    assert f1 == f1b
    assert hash(f1) == hash(f1b)
    assert tokenize(f1) == tokenize(f1b)
    f3 = SubgraphCallable(dsk2, 'c', ['d', 'f'], name=f1.name)
    assert f3 != f1
    assert hash(f3) != hash(f1)
    assert tokenize(f3) != tokenize(f1)
    f4 = SubgraphCallable(dsk2, 'a', ['d', 'e'], name=f1.name)
    assert f4 != f1
    assert hash(f4) != hash(f1)
    assert tokenize(f4) != tokenize(f1)
    f5 = SubgraphCallable(dsk1, 'c', ['e', 'd'], name=f1.name)
    assert f1 == f5
    assert hash(f1) == hash(f5)
    assert tokenize(f1) != tokenize(f5)
    f6 = SubgraphCallable(dsk1, 'c', ['d', 'e'], name='first')
    f7 = SubgraphCallable(dsk1, 'c', ['d', 'e'], name='second')
    assert f6 != f7
    assert hash(f6) != hash(f7)
    assert tokenize(f6) != tokenize(f7)
    f8 = SubgraphCallable({'a': object(), 'b': 'a'}, 'b', [], name='n')
    f9 = SubgraphCallable({'a': object(), 'b': 'a'}, 'b', [], name='n')
    assert f8 == f9
    assert hash(f8) == hash(f9)
    assert tokenize(f8) != tokenize(f9)