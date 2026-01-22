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
def test_deterministic_name():
    func = delayed(identity, pure=True)
    data1 = {'x': 1, 'y': 25, 'z': [1, 2, 3]}
    data2 = {'x': 1, 'y': 25, 'z': [1, 2, 3]}
    assert func(data1)._key == func(data2)._key