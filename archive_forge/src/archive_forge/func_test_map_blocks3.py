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
def test_map_blocks3():
    x = np.arange(10)
    y = np.arange(10) * 2
    d = da.from_array(x, chunks=5)
    e = da.from_array(y, chunks=5)
    assert_eq(da.core.map_blocks(lambda a, b: a + 2 * b, d, e, dtype=d.dtype), x + 2 * y)
    z = np.arange(100).reshape((10, 10))
    f = da.from_array(z, chunks=5)
    func = lambda a, b: a + 2 * b
    res = da.core.map_blocks(func, d, f, dtype=d.dtype)
    assert_eq(res, x + 2 * z)
    assert same_keys(da.core.map_blocks(func, d, f, dtype=d.dtype), res)
    assert_eq(da.map_blocks(func, f, d, dtype=d.dtype), z + 2 * x)