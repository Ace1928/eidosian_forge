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
@pytest.mark.parametrize('index, value', [[Ellipsis, -1], [slice(2, 8, 2), -2], [slice(8, None, 2), -3], [slice(8, None, 2), [-30]], [slice(1, None, -2), -4], [slice(1, None, -2), [-40]], [slice(3, None, 2), -5], [slice(-3, None, -2), -6], [slice(1, None, -2), -4], [slice(3, None, 2), -5], [slice(3, None, 2), [10, 11, 12, 13]], [slice(-4, None, -2), [14, 15, 16, 17]]])
def test_setitem_extended_API_1d(index, value):
    x = np.arange(10)
    dx = da.from_array(x, chunks=(4, 6))
    dx[index] = value
    x[index] = value
    assert_eq(x, dx.compute())