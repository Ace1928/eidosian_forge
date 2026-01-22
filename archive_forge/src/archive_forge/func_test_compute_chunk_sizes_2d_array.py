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
def test_compute_chunk_sizes_2d_array():
    X = np.linspace(-1, 1, num=9 * 4).reshape(9, 4)
    X = da.from_array(X, chunks=(3, 4))
    idx = X.sum(axis=1) > 0
    Y = X[idx]
    assert np.isnan(Y.shape[0]) and Y.shape[1] == 4
    assert Y.chunks == ((np.nan, np.nan, np.nan), (4,))
    Z = Y.compute_chunk_sizes()
    assert Y is Z
    assert Z.chunks == ((0, 1, 3), (4,))
    assert Z.shape == (4, 4)