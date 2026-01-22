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
def test_attribute_of_attribute():
    x = delayed(123)
    assert isinstance(x.a, Delayed)
    assert isinstance(x.a.b, Delayed)
    assert isinstance(x.a.b.c, Delayed)