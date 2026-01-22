from __future__ import annotations
import itertools
import pickle
from functools import partial
import pytest
import dask
from dask.base import tokenize
from dask.core import get_dependencies
from dask.local import get_sync
from dask.optimization import (
from dask.utils import apply, partial_by_order
from dask.utils_test import add, inc
def test_inline_functions():
    x, y, i, d = 'xyid'
    dsk = {'out': (add, i, d), i: (inc, x), d: (double, y), x: 1, y: 1}
    result = inline_functions(dsk, [], fast_functions={inc})
    expected = {'out': (add, (inc, x), d), d: (double, y), x: 1, y: 1}
    assert result == expected