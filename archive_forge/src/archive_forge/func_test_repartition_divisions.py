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
def test_repartition_divisions():
    result = repartition_divisions([0, 6], [0, 6, 6], 'a', 'b', 'c')
    assert result == {('b', 0): (methods.boundary_slice, ('a', 0), 0, 6, False), ('b', 1): (methods.boundary_slice, ('a', 0), 6, 6, True), ('c', 0): ('b', 0), ('c', 1): ('b', 1)}
    result = repartition_divisions([1, 3, 7], [1, 4, 6, 7], 'a', 'b', 'c')
    assert result == {('b', 0): (methods.boundary_slice, ('a', 0), 1, 3, False), ('b', 1): (methods.boundary_slice, ('a', 1), 3, 4, False), ('b', 2): (methods.boundary_slice, ('a', 1), 4, 6, False), ('b', 3): (methods.boundary_slice, ('a', 1), 6, 7, True), ('c', 0): (methods.concat, [('b', 0), ('b', 1)]), ('c', 1): ('b', 2), ('c', 2): ('b', 3)}