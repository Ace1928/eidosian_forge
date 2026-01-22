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
def test_map_blocks_block_info_with_broadcast():
    expected0 = [{'shape': (3, 4), 'num-chunks': (1, 2), 'array-location': [(0, 3), (0, 2)], 'chunk-location': (0, 0)}, {'shape': (3, 4), 'num-chunks': (1, 2), 'array-location': [(0, 3), (2, 4)], 'chunk-location': (0, 1)}]
    expected1 = [{'shape': (6, 2), 'num-chunks': (2, 1), 'array-location': [(0, 3), (0, 2)], 'chunk-location': (0, 0)}, {'shape': (6, 2), 'num-chunks': (2, 1), 'array-location': [(3, 6), (0, 2)], 'chunk-location': (1, 0)}]
    expected2 = [{'shape': (4,), 'num-chunks': (2,), 'array-location': [(0, 2)], 'chunk-location': (0,)}, {'shape': (4,), 'num-chunks': (2,), 'array-location': [(2, 4)], 'chunk-location': (1,)}]
    expected = [{0: expected0[0], 1: expected1[0], 2: expected2[0], None: {'shape': (6, 4), 'num-chunks': (2, 2), 'dtype': np.float64, 'chunk-shape': (3, 2), 'array-location': [(0, 3), (0, 2)], 'chunk-location': (0, 0)}}, {0: expected0[1], 1: expected1[0], 2: expected2[1], None: {'shape': (6, 4), 'num-chunks': (2, 2), 'dtype': np.float64, 'chunk-shape': (3, 2), 'array-location': [(0, 3), (2, 4)], 'chunk-location': (0, 1)}}, {0: expected0[0], 1: expected1[1], 2: expected2[0], None: {'shape': (6, 4), 'num-chunks': (2, 2), 'dtype': np.float64, 'chunk-shape': (3, 2), 'array-location': [(3, 6), (0, 2)], 'chunk-location': (1, 0)}}, {0: expected0[1], 1: expected1[1], 2: expected2[1], None: {'shape': (6, 4), 'num-chunks': (2, 2), 'dtype': np.float64, 'chunk-shape': (3, 2), 'array-location': [(3, 6), (2, 4)], 'chunk-location': (1, 1)}}]

    def func(x, y, z, block_info=None):
        for info in expected:
            if block_info[None]['chunk-location'] == info[None]['chunk-location']:
                assert block_info == info
                break
        else:
            assert False
        return x + y + z
    a = da.ones((3, 4), chunks=(3, 2))
    b = da.ones((6, 2), chunks=(3, 2))
    c = da.ones((4,), chunks=(2,))
    d = da.map_blocks(func, a, b, c, chunks=((3, 3), (2, 2)), dtype=a.dtype)
    assert d.chunks == ((3, 3), (2, 2))
    assert_eq(d, 3 * np.ones((6, 4)))