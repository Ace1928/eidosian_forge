from __future__ import annotations
import gc
import math
import os
import random
import warnings
import weakref
from bz2 import BZ2File
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from gzip import GzipFile
from itertools import repeat
import partd
import pytest
from tlz import groupby, identity, join, merge, pluck, unique, valmap
import dask
import dask.bag as db
from dask.bag.core import (
from dask.bag.utils import assert_eq
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.typing import Graph
from dask.utils import filetexts, tmpdir, tmpfile
from dask.utils_test import add, hlg_layer, hlg_layer_topological, inc
def test_reduction_with_sparse_matrices():
    sp = pytest.importorskip('scipy.sparse')
    b = db.from_sequence([sp.csr_matrix([0]) for x in range(4)], partition_size=2)

    def sp_reduce(a, b):
        return sp.vstack([a, b])
    assert b.fold(sp_reduce, sp_reduce).compute(scheduler='sync').shape == (4, 1)