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
def test_to_dataframe_optimize_graph():
    dd = pytest.importorskip('dask.dataframe')
    from dask.dataframe.utils import assert_eq as assert_eq_df
    from dask.dataframe.utils import pyarrow_strings_enabled
    x = db.from_sequence([{'name': 'test1', 'v1': 1}, {'name': 'test2', 'v1': 2}], npartitions=2)
    with dask.annotate(foo=True):
        y = x.map(lambda a: dict(**a, v2=a['v1'] + 1))
        y = y.map(lambda a: dict(**a, v3=a['v2'] + 1))
        y = y.map(lambda a: dict(**a, v4=a['v3'] + 1))
    assert len(y.dask) == y.npartitions * 4
    d = y.to_dataframe()
    if not dd._dask_expr_enabled():
        assert len(d.dask) < len(y.dask) + d.npartitions * int(pyarrow_strings_enabled())
    d2 = y.to_dataframe(optimize_graph=False)
    if not dd._dask_expr_enabled():
        assert len(d2.dask.keys() - y.dask.keys()) == d.npartitions * (1 + int(pyarrow_strings_enabled()))
    if not dd._dask_expr_enabled():
        assert hlg_layer_topological(d2.dask, 1).annotations == {'foo': True}
    assert_eq_df(d, d2)