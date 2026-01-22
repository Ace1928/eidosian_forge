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
@pytest.mark.parametrize('x', [[1, 2], (1, 2), (add, 1, 2), [], ()])
def test_nout_with_tasks(x):
    length = len(x)
    d = delayed(x, nout=length)
    assert len(d) == len(list(d)) == length
    assert d.compute() == x