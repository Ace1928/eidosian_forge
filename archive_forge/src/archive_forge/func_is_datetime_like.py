from __future__ import annotations
import functools
from typing import Any
import numpy as np
from xarray.core import utils
def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types"""
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)