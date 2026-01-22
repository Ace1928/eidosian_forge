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
def test_block_nested():
    a1 = np.array([1, 1, 1])
    a2 = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    a3 = np.array([3, 3, 3])
    a4 = np.array([4, 4, 4])
    a5 = np.array(5)
    a6 = np.array([6, 6, 6, 6, 6])
    a7 = np.zeros((2, 6))
    d1 = da.asarray(a1)
    d2 = da.asarray(a2)
    d3 = da.asarray(a3)
    d4 = da.asarray(a4)
    d5 = da.asarray(a5)
    d6 = da.asarray(a6)
    d7 = da.asarray(a7)
    expected = np.block([[np.block([[a1], [a3], [a4]]), a2], [a5, a6], [a7]])
    result = da.block([[da.block([[d1], [d3], [d4]]), d2], [d5, d6], [d7]])
    assert_eq(expected, result)