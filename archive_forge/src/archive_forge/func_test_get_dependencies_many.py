from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_get_dependencies_many():
    dsk = {'a': [1, 2, 3], 'b': 'a', 'c': [1, (inc, 1)], 'd': [(sum, 'c')], 'e': ['a', 'b', 'zzz'], 'f': [['a', 'b'], 2, 3]}
    tasks = [dsk[k] for k in ('d', 'f')]
    s = get_dependencies(dsk, task=tasks)
    assert s == {'a', 'b', 'c'}
    s = get_dependencies(dsk, task=tasks, as_list=True)
    assert sorted(s) == ['a', 'b', 'c']
    s = get_dependencies(dsk, task=[])
    assert s == set()
    s = get_dependencies(dsk, task=[], as_list=True)
    assert s == []