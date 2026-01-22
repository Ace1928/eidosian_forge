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
def test_foldby_tree_reduction():
    dsk = list()
    for n in [1, 7, 32]:
        b = db.from_sequence(range(100), npartitions=n)
        c = b.foldby(iseven, add)
        dsk.extend([c])
        for m in [False, None, 2, 3]:
            d = b.foldby(iseven, add, split_every=m)
            e = b.foldby(iseven, add, 0, split_every=m)
            f = b.foldby(iseven, add, 0, add, split_every=m)
            g = b.foldby(iseven, add, 0, add, 0, split_every=m)
            dsk.extend([d, e, f, g])
    results = dask.compute(dsk)
    first = results[0]
    assert all([r == first for r in results])