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
def test_multiple_repartition_partition_size():
    b = db.from_sequence(range(1, 100), npartitions=1)
    total_mem = sum(b.map_partitions(total_mem_usage).compute())
    c = b.repartition(partition_size=total_mem // 2)
    assert c.npartitions >= 2
    assert_eq(b, c)
    d = c.repartition(partition_size=total_mem // 5)
    assert d.npartitions >= 5
    assert_eq(c, d)