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
def test_parse_bytes():
    assert parse_bytes('100') == 100
    assert parse_bytes('100 MB') == 100000000
    assert parse_bytes('100M') == 100000000
    assert parse_bytes('5kB') == 5000
    assert parse_bytes('5.4 kB') == 5400
    assert parse_bytes('1kiB') == 1024
    assert parse_bytes('1Mi') == 2 ** 20
    assert parse_bytes('1e6') == 1000000
    assert parse_bytes('1e6 kB') == 1000000000
    assert parse_bytes('MB') == 1000000
    assert parse_bytes(123) == 123
    assert parse_bytes('.5GB') == 500000000