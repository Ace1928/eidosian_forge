from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
from importlib import import_module
import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import (  # noqa
from numpy import concatenate as _concatenate
from numpy.lib.stride_tricks import sliding_window_view  # noqa
from packaging.version import Version
from xarray.core import dask_array_ops, dtypes, nputils
from xarray.core.options import OPTIONS
from xarray.core.utils import is_duck_array, is_duck_dask_array, module_available
from xarray.namedarray import pycompat
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, is_chunked_array
def timedelta_to_numeric(value, datetime_unit='ns', dtype=float):
    """Convert a timedelta-like object to numerical values.

    Parameters
    ----------
    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str
        Time delta representation.
    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
        The time units of the output values. Note that some conversions are not allowed due to
        non-linear relationships between units.
    dtype : type
        The output data type.

    """
    import datetime as dt
    if isinstance(value, dt.timedelta):
        out = py_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, np.timedelta64):
        out = np_timedelta64_to_float(value, datetime_unit)
    elif isinstance(value, pd.Timedelta):
        out = pd_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, str):
        try:
            a = pd.to_timedelta(value)
        except ValueError:
            raise ValueError(f'Could not convert {value!r} to timedelta64 using pandas.to_timedelta')
        return py_timedelta_to_float(a, datetime_unit)
    else:
        raise TypeError(f'Expected value of type str, pandas.Timedelta, datetime.timedelta or numpy.timedelta64, but received {type(value).__name__}')
    return out.astype(dtype)