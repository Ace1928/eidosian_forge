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
def test_custom_delayed():
    x = Tuple({'a': 1, 'b': 2, 'c': (add, 'a', 'b')}, ['a', 'b', 'c'])
    x2 = delayed(add, pure=True)(x, (4, 5, 6))
    n = delayed(len, pure=True)(x)
    assert delayed(len, pure=True)(x).key == n.key
    assert x2.compute() == (1, 2, 3, 4, 5, 6)
    assert compute(n, x2, x) == (3, (1, 2, 3, 4, 5, 6), (1, 2, 3))