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
def test_random_partitions():
    a, b = d.random_split([0.5, 0.5], 42)
    assert isinstance(a, dd.DataFrame)
    assert isinstance(b, dd.DataFrame)
    assert a._name != b._name
    np.testing.assert_array_equal(a.index, sorted(a.index))
    assert len(a.compute()) + len(b.compute()) == len(full)
    a2, b2 = d.random_split([0.5, 0.5], 42)
    assert a2._name == a._name
    assert b2._name == b._name
    a, b = d.random_split([0.5, 0.5], 42, True)
    a2, b2 = d.random_split([0.5, 0.5], 42, True)
    assert_eq(a, a2)
    assert_eq(b, b2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(a.index, sorted(a.index))
    parts = d.random_split([0.4, 0.5, 0.1], 42)
    names = {p._name for p in parts}
    names.update([a._name, b._name])
    assert len(names) == 5
    with pytest.raises(ValueError):
        d.random_split([0.4, 0.5], 42)