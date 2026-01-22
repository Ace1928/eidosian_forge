from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_funcname_long():

    def a_long_function_name_11111111111111111111111111111111111111111111111():
        pass
    result = funcname(a_long_function_name_11111111111111111111111111111111111111111111111)
    assert 'a_long_function_name' in result
    assert len(result) < 60