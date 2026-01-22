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
def test_setitem_extended_API_2d_rhs_func_of_lhs():
    x = np.arange(60).reshape((6, 10))
    chunks = (2, 3)
    dx = da.from_array(x, chunks=chunks)
    dx[2:4, dx[0] > 3] = -5
    x[2:4, x[0] > 3] = -5
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[2, dx[0] < -2] = -7
    x[2, x[0] < -2] = -7
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[dx % 2 == 0] = -8
    x[x % 2 == 0] = -8
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[dx % 2 == 0] = -8
    x[x % 2 == 0] = -8
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[3:5, 5:1:-2] = -dx[:2, 4:1:-2]
    x[3:5, 5:1:-2] = -x[:2, 4:1:-2]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[0, 1:3] = -dx[0, 4:2:-1]
    x[0, 1:3] = -x[0, 4:2:-1]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[...] = dx
    x[...] = x
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[...] = dx[...]
    x[...] = x[...]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[0] = dx[-1]
    x[0] = x[-1]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[0, :] = dx[-2, :]
    x[0, :] = x[-2, :]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[:, 1] = dx[:, -3]
    x[:, 1] = x[:, -3]
    assert_eq(x, dx.compute())
    index = da.from_array([0, 2], chunks=(2,))
    dx = da.from_array(x, chunks=chunks)
    dx[index, 8] = [99, 88]
    x[[0, 2], 8] = [99, 88]
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=chunks)
    dx[:, index] = dx[:, :2]
    x[:, [0, 2]] = x[:, :2]
    assert_eq(x, dx.compute())
    index = da.where(da.arange(3, chunks=(1,)) < 2)[0]
    dx = da.from_array(x, chunks=chunks)
    dx[index, 7] = [-23, -33]
    x[index.compute(), 7] = [-23, -33]
    assert_eq(x, dx.compute())
    index = da.where(da.arange(3, chunks=(1,)) < 2)[0]
    dx = da.from_array(x, chunks=chunks)
    dx[index,] = -34
    x[index.compute(),] = -34
    assert_eq(x, dx.compute())
    index = index - 4
    dx = da.from_array(x, chunks=chunks)
    dx[index, 7] = [-43, -53]
    x[index.compute(), 7] = [-43, -53]
    assert_eq(x, dx.compute())
    index = da.from_array([0, -1], chunks=(1,))
    x[[0, -1]] = 9999
    dx[index,] = 9999
    assert_eq(x, dx.compute())
    dx = da.from_array(x, chunks=(-1, -1))
    dx[...] = da.from_array(x, chunks=chunks)
    assert_eq(x, dx.compute())
    dx = da.from_array(x.copy(), chunks=(2, 3))
    v = x.reshape((1, 1) + x.shape)
    x[...] = v
    dx[...] = v
    assert_eq(x, dx.compute())
    index = da.where(da.arange(3, chunks=(1,)) < 2)[0]
    v = -np.arange(12).reshape(1, 1, 6, 2)
    x[:, [0, 1]] = v
    dx[:, index] = v
    assert_eq(x, dx.compute())