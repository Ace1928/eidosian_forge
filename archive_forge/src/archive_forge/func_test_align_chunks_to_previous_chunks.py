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
def test_align_chunks_to_previous_chunks():
    chunks = normalize_chunks('auto', shape=(2000,), previous_chunks=(512,), limit='600 B', dtype=np.uint8)
    assert chunks == ((512, 512, 512, 2000 - 512 * 3),)
    chunks = normalize_chunks('auto', shape=(2000,), previous_chunks=(128,), limit='600 B', dtype=np.uint8)
    assert chunks == ((512, 512, 512, 2000 - 512 * 3),)
    chunks = normalize_chunks('auto', shape=(2000,), previous_chunks=(512,), limit='1200 B', dtype=np.uint8)
    assert chunks == ((1024, 2000 - 1024),)
    chunks = normalize_chunks('auto', shape=(3, 10211, 10376), previous_chunks=(1, 512, 512), limit='1MiB', dtype=np.float32)
    assert chunks[0] == (1, 1, 1)
    assert all((c % 512 == 0 for c in chunks[1][:-1]))
    assert all((c % 512 == 0 for c in chunks[2][:-1]))