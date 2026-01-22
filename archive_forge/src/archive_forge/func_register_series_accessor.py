from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def register_series_accessor(name):
    """
    Register a custom accessor on :class:`dask.dataframe.Series`.

    See :func:`pandas.api.extensions.register_series_accessor` for more.
    """
    from dask.dataframe import Series
    return _register_accessor(name, Series)