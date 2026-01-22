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
def test_elemwise_uneven_chunks():
    rng = da.random.default_rng()
    x = da.arange(10, chunks=((4, 6),))
    y = da.ones(10, chunks=((6, 4),))
    assert_eq(x + y, np.arange(10) + np.ones(10))
    z = x + y
    assert z.chunks == ((4, 2, 4),)
    x = rng.random((10, 10), chunks=((4, 6), (5, 2, 3)))
    y = rng.random((4, 10, 10), chunks=((2, 2), (6, 4), (2, 3, 5)))
    z = x + y
    assert_eq(x + y, x.compute() + y.compute())
    assert z.chunks == ((2, 2), (4, 2, 4), (2, 3, 2, 3))