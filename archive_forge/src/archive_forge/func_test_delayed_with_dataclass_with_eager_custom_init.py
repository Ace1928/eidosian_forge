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
def test_delayed_with_dataclass_with_eager_custom_init():

    @dataclass()
    class ADataClass:
        a: int

        def __init__(self, b: int):
            self.a = b
    with_class = delayed({'data': ADataClass(b=3)})

    def return_nested(obj):
        return obj['data'].a
    final = delayed(return_nested)(with_class)
    assert final.compute() == 3