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
@pytest.mark.parametrize('axis', [0, 1])
def test_scipy_sparse_concatenate(axis):
    pytest.importorskip('scipy.sparse')
    import scipy.sparse
    rng = da.random.default_rng()
    xs = []
    ys = []
    for _ in range(2):
        x = rng.random((1000, 10), chunks=(100, 10))
        x[x < 0.9] = 0
        xs.append(x)
        ys.append(x.map_blocks(scipy.sparse.csr_matrix))
    z = da.concatenate(ys, axis=axis)
    z = z.compute()
    if axis == 0:
        sp_concatenate = scipy.sparse.vstack
    elif axis == 1:
        sp_concatenate = scipy.sparse.hstack
    z_expected = sp_concatenate([scipy.sparse.csr_matrix(e.compute()) for e in xs])
    assert (z != z_expected).nnz == 0