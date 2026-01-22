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
@pytest.mark.parametrize('nworkers', [100, 250, 500, 1000])
def test_default_partitioning_worker_saturation(nworkers):
    ntasks = 0
    nitems = 1
    while ntasks < nworkers:
        ntasks = len(db.from_sequence(range(nitems)).dask)
        nitems += math.floor(max(1, nworkers / 10))
        assert nitems < 20000