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
def test_map_partitions_arg():

    def append_str(partition, s):
        return [x + s for x in partition]
    mybag = db.from_sequence(['a', 'b', 'c'])
    assert_eq(mybag.map_partitions(append_str, 'foo'), ['afoo', 'bfoo', 'cfoo'])
    assert_eq(mybag.map_partitions(append_str, dask.delayed('foo')), ['afoo', 'bfoo', 'cfoo'])