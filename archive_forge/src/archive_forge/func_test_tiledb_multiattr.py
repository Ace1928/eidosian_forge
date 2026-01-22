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
def test_tiledb_multiattr():
    tiledb = pytest.importorskip('tiledb')
    dom = tiledb.Domain(tiledb.Dim('x', (0, 1000), tile=100), tiledb.Dim('y', (0, 1000), tile=100))
    schema = tiledb.ArraySchema(attrs=(tiledb.Attr('attr1'), tiledb.Attr('attr2')), domain=dom)
    with tmpdir() as uri:
        tiledb.DenseArray.create(uri, schema)
        tdb = tiledb.DenseArray(uri, 'w')
        rng = np.random.default_rng()
        ar1 = rng.standard_normal(tdb.schema.shape)
        ar2 = rng.standard_normal(tdb.schema.shape)
        tdb[:] = {'attr1': ar1, 'attr2': ar2}
        tdb = tiledb.DenseArray(uri, 'r')
        d = da.from_tiledb(uri, attribute='attr2')
        assert_eq(d, ar2)
        d = da.from_tiledb(uri, attribute='attr2')
        assert_eq(np.mean(ar2), d.mean().compute(scheduler='threads'))