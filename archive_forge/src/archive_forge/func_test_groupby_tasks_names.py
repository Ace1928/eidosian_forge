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
def test_groupby_tasks_names():
    b = db.from_sequence(range(160), npartitions=4)
    func = lambda x: x % 10
    func2 = lambda x: x % 20
    assert set(b.groupby(func, max_branch=4, shuffle='tasks').dask) == set(b.groupby(func, max_branch=4, shuffle='tasks').dask)
    assert set(b.groupby(func, max_branch=4, shuffle='tasks').dask) != set(b.groupby(func, max_branch=2, shuffle='tasks').dask)
    assert set(b.groupby(func, max_branch=4, shuffle='tasks').dask) != set(b.groupby(func2, max_branch=4, shuffle='tasks').dask)