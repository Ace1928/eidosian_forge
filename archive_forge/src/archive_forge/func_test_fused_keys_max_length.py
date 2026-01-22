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
def test_fused_keys_max_length():
    d = {'u-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (inc, 'v-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong'), 'v-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (inc, 'w-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong'), 'w-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (inc, 'x-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong'), 'x-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (inc, 'y-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong'), 'y-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (inc, 'z-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong'), 'z-looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong': (add, 'a', 'b'), 'a': 1, 'b': 2}
    fused, deps = fuse(d, rename_keys=True)
    for key in fused:
        assert len(key) < 150