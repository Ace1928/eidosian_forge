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
def test_inline_functions_protects_output_keys():
    dsk = {'x': (inc, 1), 'y': (double, 'x')}
    assert inline_functions(dsk, [], [inc]) == {'y': (double, (inc, 1))}
    assert inline_functions(dsk, ['x'], [inc]) == {'y': (double, 'x'), 'x': (inc, 1)}