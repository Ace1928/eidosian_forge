from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def isdatetime(value) -> bool:
    """
    Whether the array or scalar is recognized datetime type.
    """
    if is_series(value) and len(value):
        return isinstance(value.iloc[0], datetime_types)
    elif isinstance(value, np.ndarray):
        return value.dtype.kind == 'M' or (value.dtype.kind == 'O' and len(value) != 0 and isinstance(value[0], datetime_types))
    elif isinstance(value, list):
        return all((isinstance(d, datetime_types) for d in value))
    else:
        return isinstance(value, datetime_types)