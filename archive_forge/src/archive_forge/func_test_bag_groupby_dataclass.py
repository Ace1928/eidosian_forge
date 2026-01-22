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
@pytest.mark.parametrize('key', [Key(foo=1), Key(foo=1, bar=2)], ids=['none_field', 'no_none_fields'])
@pytest.mark.parametrize('shuffle', ['disk', 'tasks'])
@pytest.mark.parametrize('scheduler', ['synchronous', 'processes'])
def test_bag_groupby_dataclass(key, shuffle, scheduler):
    seq = [(key, i) for i in range(50)]
    b = db.from_sequence(seq).groupby(lambda x: x[0], shuffle=shuffle)
    with dask.config.set(scheduler=scheduler):
        result = b.compute()
    assert len(result) == 1