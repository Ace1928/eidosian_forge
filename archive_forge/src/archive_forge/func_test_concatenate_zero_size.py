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
def test_concatenate_zero_size():
    x = np.random.default_rng().random(10)
    y = da.from_array(x, chunks=3)
    result_np = np.concatenate([x, x[:0]])
    result_da = da.concatenate([y, y[:0]])
    assert_eq(result_np, result_da)
    assert result_da is y
    result_np = np.concatenate([np.zeros(0, dtype=float), np.zeros(1, dtype=int)])
    result_da = da.concatenate([da.zeros(0, dtype=float), da.zeros(1, dtype=int)])
    assert_eq(result_np, result_da)
    result_np = np.concatenate([np.zeros(0), np.zeros(0)])
    result_da = da.concatenate([da.zeros(0), da.zeros(0)])
    assert_eq(result_np, result_da)