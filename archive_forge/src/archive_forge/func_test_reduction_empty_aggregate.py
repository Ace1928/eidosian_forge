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
@pytest.mark.parametrize('npartitions', [1, 2, 4])
def test_reduction_empty_aggregate(npartitions):
    b = db.from_sequence([0, 0, 0, 1], npartitions=npartitions).filter(None)
    assert_eq(b.min(split_every=2), 1)
    vals = db.compute(b.min(split_every=2), b.max(split_every=2), scheduler='sync')
    assert vals == (1, 1)
    with pytest.raises(ValueError):
        b = db.from_sequence([0, 0, 0, 0], npartitions=npartitions)
        b.filter(None).min(split_every=2).compute(scheduler='sync')