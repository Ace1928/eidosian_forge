from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def datetime_column_to_ndarray(col: Column) -> tuple[np.ndarray | pd.Series, Any]:
    """
    Convert a column holding DateTime data to a NumPy array.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    buffers = col.get_buffers()
    _, col_bit_width, format_str, _ = col.dtype
    dbuf, _ = buffers['data']
    data = buffer_to_ndarray(dbuf, (DtypeKind.INT, col_bit_width, getattr(ArrowCTypes, f'INT{col_bit_width}'), Endianness.NATIVE), offset=col.offset, length=col.size())
    data = parse_datetime_format_str(format_str, data)
    data = set_nulls(data, col, buffers['validity'])
    return (data, buffers)