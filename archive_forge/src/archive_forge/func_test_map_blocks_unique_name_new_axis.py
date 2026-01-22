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
def test_map_blocks_unique_name_new_axis():

    def func(some_2d, block_info=None):
        if not block_info:
            return some_2d
        dtype = block_info[None]['dtype']
        return np.zeros(block_info[None]['shape'], dtype=dtype)
    input_arr = da.zeros((3, 4), chunks=((3,), (4,)), dtype=np.float32)
    x = da.map_blocks(func, input_arr, new_axis=[0], dtype=np.float32)
    assert x.chunks == ((1,), (3,), (4,))
    assert_eq(x, np.zeros((1, 3, 4), dtype=np.float32))
    y = da.map_blocks(func, input_arr, new_axis=[2], dtype=np.float32)
    assert y.chunks == ((3,), (4,), (1,))
    assert_eq(y, np.zeros((3, 4, 1), dtype=np.float32))
    assert x.name != y.name