from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def assert_array_index_eq(left, right, check_divisions=False):
    """left and right are equal, treating index and array as equivalent"""
    assert_eq(left, pd.Index(right) if isinstance(right, np.ndarray) else right, check_divisions=check_divisions)