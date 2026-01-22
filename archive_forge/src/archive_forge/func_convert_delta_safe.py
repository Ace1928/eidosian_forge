from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def convert_delta_safe(base, deltas, unit) -> Series:
    """
        Convert base dates and deltas to datetimes, using pandas vectorized
        versions if the deltas satisfy restrictions required to be expressed
        as dates in pandas.
        """
    index = getattr(deltas, 'index', None)
    if unit == 'd':
        if deltas.max() > MAX_DAY_DELTA or deltas.min() < MIN_DAY_DELTA:
            values = [base + timedelta(days=int(d)) for d in deltas]
            return Series(values, index=index)
    elif unit == 'ms':
        if deltas.max() > MAX_MS_DELTA or deltas.min() < MIN_MS_DELTA:
            values = [base + timedelta(microseconds=int(d) * 1000) for d in deltas]
            return Series(values, index=index)
    else:
        raise ValueError('format not understood')
    base = to_datetime(base)
    deltas = to_timedelta(deltas, unit=unit)
    return base + deltas