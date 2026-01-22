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
def test_map_blocks_block_info_with_new_axis():
    values = da.from_array(np.array(['a', 'a', 'b', 'c']), 2)

    def func(x, block_info=None):
        assert block_info.keys() == {0, None}
        assert block_info[0]['shape'] == (4,)
        assert block_info[0]['num-chunks'] == (2,)
        assert block_info[None]['shape'] == (4, 3)
        assert block_info[None]['num-chunks'] == (2, 1)
        assert block_info[None]['chunk-shape'] == (2, 3)
        assert block_info[None]['dtype'] == np.dtype('f8')
        assert block_info[0]['chunk-location'] in {(0,), (1,)}
        if block_info[0]['chunk-location'] == (0,):
            assert block_info[0]['array-location'] == [(0, 2)]
            assert block_info[None]['chunk-location'] == (0, 0)
            assert block_info[None]['array-location'] == [(0, 2), (0, 3)]
        elif block_info[0]['chunk-location'] == (1,):
            assert block_info[0]['array-location'] == [(2, 4)]
            assert block_info[None]['chunk-location'] == (1, 0)
            assert block_info[None]['array-location'] == [(2, 4), (0, 3)]
        return np.ones((len(x), 3))
    z = values.map_blocks(func, chunks=((2, 2), 3), new_axis=1, dtype='f8')
    assert_eq(z, np.ones((4, 3), dtype='f8'))