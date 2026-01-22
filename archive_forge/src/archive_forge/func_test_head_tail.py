from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
def test_head_tail():
    assert_eq(d.head(2), full.head(2))
    assert_eq(d.head(3), full.head(3))
    assert_eq(d.head(2), dsk['x', 0].head(2))
    assert_eq(d['a'].head(2), full['a'].head(2))
    assert_eq(d['a'].head(3), full['a'].head(3))
    assert_eq(d['a'].head(2), dsk['x', 0]['a'].head(2))
    assert sorted(d.head(2, compute=False).dask) == sorted(d.head(2, compute=False).dask)
    assert sorted(d.head(2, compute=False).dask) != sorted(d.head(3, compute=False).dask)
    assert_eq(d.tail(2), full.tail(2))
    assert_eq(d.tail(3), full.tail(3))
    assert_eq(d.tail(2), dsk['x', 2].tail(2))
    assert_eq(d['a'].tail(2), full['a'].tail(2))
    assert_eq(d['a'].tail(3), full['a'].tail(3))
    assert_eq(d['a'].tail(2), dsk['x', 2]['a'].tail(2))
    assert sorted(d.tail(2, compute=False).dask) == sorted(d.tail(2, compute=False).dask)
    assert sorted(d.tail(2, compute=False).dask) != sorted(d.tail(3, compute=False).dask)