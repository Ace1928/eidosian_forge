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
def test_read_text_large_gzip():
    with tmpfile('gz') as fn:
        data = b'Hello, world!\n' * 100
        f = GzipFile(fn, 'wb')
        f.write(data)
        f.close()
        with pytest.raises(ValueError):
            db.read_text(fn, blocksize=50, linedelimiter='\n')
        c = db.read_text(fn, blocksize=None)
        assert c.npartitions == 1
        assert ''.join(c.compute()) == data.decode()