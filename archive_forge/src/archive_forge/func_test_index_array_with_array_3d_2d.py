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
@pytest.mark.xfail(reason='Chunking does not align well')
def test_index_array_with_array_3d_2d():
    x = np.arange(4 ** 3).reshape((4, 4, 4))
    dx = da.from_array(x, chunks=(2, 2, 2))
    ind = np.random.default_rng().random((4, 4)) > 0.5
    ind = np.arange(4 ** 2).reshape((4, 4)) % 2 == 0
    dind = da.from_array(ind, (2, 2))
    assert_eq(x[ind], dx[dind])
    assert_eq(x[:, ind], dx[:, dind])