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
def test_map_blocks_block_info():
    x = da.arange(50, chunks=10)

    def func(a, b, c, block_info=None):
        for idx in [0, 2, None]:
            assert block_info[idx]['shape'] == (50,)
            assert block_info[idx]['num-chunks'] == (5,)
            start, stop = block_info[idx]['array-location'][0]
            assert stop - start == 10
            assert 0 <= start <= 40
            assert 10 <= stop <= 50
            assert 0 <= block_info[idx]['chunk-location'][0] <= 4
        assert block_info[None]['chunk-shape'] == (10,)
        assert block_info[None]['dtype'] == x.dtype
        return a + b + c
    z = da.map_blocks(func, x, 100, x + 1, dtype=x.dtype)
    assert_eq(z, x + x + 1 + 100)