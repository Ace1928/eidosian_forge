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
def test_take_npartitions():
    assert list(b.take(6, npartitions=2)) == [0, 1, 2, 3, 4, 0]
    assert b.take(6, npartitions=-1) == (0, 1, 2, 3, 4, 0)
    assert b.take(3, npartitions=-1) == (0, 1, 2)
    with pytest.raises(ValueError):
        b.take(1, npartitions=5)