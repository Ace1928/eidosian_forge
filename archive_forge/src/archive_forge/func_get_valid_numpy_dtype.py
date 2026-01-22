from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
def get_valid_numpy_dtype(array: np.ndarray | pd.Index):
    """Return a numpy compatible dtype from either
    a numpy array or a pandas.Index.

    Used for wrapping a pandas.Index as an xarray,Variable.

    """
    if isinstance(array, pd.PeriodIndex):
        dtype = np.dtype('O')
    elif hasattr(array, 'categories'):
        dtype = array.categories.dtype
        if not is_valid_numpy_dtype(dtype):
            dtype = np.dtype('O')
    elif not is_valid_numpy_dtype(array.dtype):
        dtype = np.dtype('O')
    else:
        dtype = array.dtype
    return dtype