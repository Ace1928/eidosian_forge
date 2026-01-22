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
def test_check_meta_flag():
    dd = pytest.importorskip('dask.dataframe')
    from pandas import Series
    a = Series(['a', 'b', 'a'], dtype='category')
    b = Series(['a', 'c', 'a'], dtype='category')
    da = delayed(lambda x: x)(a)
    db = delayed(lambda x: x)(b)
    c = dd.from_delayed([da, db], verify_meta=False)
    dd.utils.assert_eq(c, c)