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
def test_normalize_chunks():
    assert normalize_chunks(3, (4, 6)) == ((3, 1), (3, 3))
    assert normalize_chunks(((3, 3), (8,)), (6, 8)) == ((3, 3), (8,))
    assert normalize_chunks((4, 5), (9,)) == ((4, 5),)
    assert normalize_chunks((4, 5), (9, 9)) == ((4, 4, 1), (5, 4))
    assert normalize_chunks(-1, (5, 5)) == ((5,), (5,))
    assert normalize_chunks((3, -1), (5, 5)) == ((3, 2), (5,))
    assert normalize_chunks((3, None), (5, 5)) == ((3, 2), (5,))
    assert normalize_chunks({0: 3}, (5, 5)) == ((3, 2), (5,))
    assert normalize_chunks([[2, 2], [3, 3]]) == ((2, 2), (3, 3))
    assert normalize_chunks(10, (30, 5)) == ((10, 10, 10), (5,))
    assert normalize_chunks((), (0, 0)) == ((0,), (0,))
    assert normalize_chunks(-1, (0, 3)) == ((0,), (3,))
    assert normalize_chunks(((float('nan'),),)) == ((np.nan,),)
    assert normalize_chunks('auto', shape=(20,), limit=5, dtype='uint8') == ((5, 5, 5, 5),)
    assert normalize_chunks(('auto', None), (5, 5), dtype=int) == ((5,), (5,))
    with pytest.raises(ValueError):
        normalize_chunks(((10,),), (11,))
    with pytest.raises(ValueError):
        normalize_chunks(((5,), (5,)), (5,))