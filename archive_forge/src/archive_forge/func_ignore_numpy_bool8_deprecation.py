from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
@contextlib.contextmanager
def ignore_numpy_bool8_deprecation():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias for `np.bool_`')
        yield