from __future__ import annotations
import pickle
import types
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from operator import add, setitem
from random import random
from typing import NamedTuple
import cloudpickle
import pytest
from tlz import merge
import dask
import dask.bag as db
from dask import compute
from dask.delayed import Delayed, delayed, to_task_dask
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import inc
@pytest.mark.filterwarnings('ignore:The dask.delayed:UserWarning')
def test_to_task_dask():
    a = delayed(1, name='a')
    b = delayed(2, name='b')
    task, dask = to_task_dask([a, b, 3])
    assert task == ['a', 'b', 3]
    task, dask = to_task_dask((a, b, 3))
    assert task == (tuple, ['a', 'b', 3])
    assert dict(dask) == merge(a.dask, b.dask)
    task, dask = to_task_dask({a: 1, b: 2})
    assert task == (dict, [['b', 2], ['a', 1]]) or task == (dict, [['a', 1], ['b', 2]])
    assert dict(dask) == merge(a.dask, b.dask)
    f = namedtuple('f', ['a', 'b', 'c'])
    x = f(a, b, 3)
    task, dask = to_task_dask(x)
    assert task == (f, 'a', 'b', 3)
    assert dict(dask) == merge(a.dask, b.dask)
    task, dask = to_task_dask(slice(a, b, 3))
    assert task == (slice, 'a', 'b', 3)
    assert dict(dask) == merge(a.dask, b.dask)

    class MyClass(dict):
        pass
    task, dask = to_task_dask(MyClass())
    assert type(task) is MyClass
    assert dict(dask) == {}
    x = Tuple({'a': 1, 'b': 2, 'c': (add, 'a', 'b')}, ['a', 'b', 'c'])
    task, dask = to_task_dask(x)
    assert task in dask
    f = dask.pop(task)
    assert f == (tuple, ['a', 'b', 'c'])
    assert dask == x._dask