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
@pytest.mark.parametrize('typ', [list, tuple, set])
def test_iterators(typ):
    a = delayed(1)
    b = delayed(2)
    c = delayed(sum)(iter(typ([a, b])))
    x = c.compute()
    assert x == 3

    def f(seq):
        return sum(seq)
    c = delayed(f)(iter(typ([a, b])))
    assert c.compute() == 3