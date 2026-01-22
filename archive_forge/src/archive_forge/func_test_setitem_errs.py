from __future__ import annotations
import contextlib
import copy
import pathlib
import re
import xml.etree.ElementTree
from unittest import mock
import pytest
import math
import operator
import os
import time
import warnings
from functools import reduce
from io import StringIO
from operator import add, sub
from threading import Lock
from tlz import concat, merge
from tlz.curried import identity
import dask
import dask.array as da
from dask.array.chunk import getitem
from dask.array.core import (
from dask.array.numpy_compat import NUMPY_GE_200
from dask.array.reshape import _not_implemented_message
from dask.array.tests.test_dispatch import EncapsulateNDArray
from dask.array.utils import assert_eq, same_keys
from dask.base import compute_as_if_collection, tokenize
from dask.blockwise import broadcast_dimensions
from dask.blockwise import make_blockwise_graph as top
from dask.blockwise import optimize_blockwise
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.layers import Blockwise
from dask.utils import SerializableLock, key_split, parse_bytes, tmpdir, tmpfile
from dask.utils_test import dec, hlg_layer_topological, inc
def test_setitem_errs():
    x = da.ones((4, 4), chunks=(2, 2))
    with pytest.raises(ValueError):
        x[x > 1] = x
    with pytest.raises(ValueError):
        x[[True, True, False, False], 0] = [2, 3, 4]
    with pytest.raises(ValueError):
        x[[True, True, True, False], 0] = [2, 3]
    with pytest.raises(ValueError):
        x[0, [True, True, True, False]] = [2, 3]
    with pytest.raises(ValueError):
        x[0, [True, True, True, False]] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        x[da.from_array([True, True, True, False]), 0] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        x[0, da.from_array([True, False, False, True])] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        x[:, 0] = [2, 3, 4]
    with pytest.raises(ValueError):
        x[0, :] = [1, 2, 3, 4, 5]
    x = da.ones((4, 4), chunks=(2, 2))
    with pytest.raises(IndexError):
        x[:, :, :] = 2
    with pytest.raises(IndexError):
        x[[[True, True, False, False]], 0] = 5
    with pytest.raises(IndexError):
        x[[True, True, False]] = 5
    with pytest.raises(IndexError):
        x[[False, True, True, True, False]] = 5
    with pytest.raises(IndexError):
        x[[[1, 2, 3]], 0] = 5
    with pytest.raises(NotImplementedError):
        x[[1, 2], [2, 3]] = 6
    with pytest.raises(NotImplementedError):
        x[[True, True, False, False], [2, 3]] = 5
    with pytest.raises(NotImplementedError):
        x[[True, True, False, False], [False, True, False, False]] = 7
    with pytest.raises(NotImplementedError):
        x[True] = 5
    with pytest.raises(NotImplementedError):
        x[np.array(True)] = 5
    with pytest.raises(NotImplementedError):
        x[0, da.from_array(True)] = 5
    y = da.from_array(np.array(1))
    with pytest.raises(IndexError):
        y[:] = 2
    x = np.arange(12).reshape((3, 4))
    dx = da.from_array(x, chunks=(2, 2))
    with pytest.raises(ValueError):
        dx[...] = np.arange(24).reshape((2, 1, 3, 4))
    dx = da.unique(da.random.default_rng().random([10]))
    with pytest.raises(ValueError, match='Arrays chunk sizes are unknown'):
        dx[0] = 0
    x = da.ones((3, 3), dtype=int)
    with pytest.raises(ValueError, match='cannot convert float NaN to integer'):
        x[:, 1] = np.nan