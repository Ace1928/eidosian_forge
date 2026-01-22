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
def test_zarr_group():
    zarr = pytest.importorskip('zarr')
    with tmpdir() as d:
        a = da.zeros((3, 3), chunks=(1, 1))
        a.to_zarr(d, component='test')
        with pytest.raises((OSError, ValueError)):
            a.to_zarr(d, component='test', overwrite=False)
        a.to_zarr(d, component='test', overwrite=True)
        a.to_zarr(d, component='test2', overwrite=False)
        a.to_zarr(d, component='nested/test', overwrite=False)
        group = zarr.open_group(d, mode='r')
        assert list(group) == ['nested', 'test', 'test2']
        assert 'test' in group['nested']
        a2 = da.from_zarr(d, component='test')
        assert_eq(a, a2)
        assert a2.chunks == a.chunks