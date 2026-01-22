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
def string_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    """
    Convert a column holding string data to a NumPy array.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    null_kind, sentinel_val = col.describe_null
    if null_kind not in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        raise NotImplementedError(f'{null_kind} null kind is not yet supported for string columns.')
    buffers = col.get_buffers()
    assert buffers['offsets'], 'String buffers must contain offsets'
    data_buff, _ = buffers['data']
    assert col.dtype[2] in (ArrowCTypes.STRING, ArrowCTypes.LARGE_STRING)
    data_dtype = (DtypeKind.UINT, 8, ArrowCTypes.UINT8, Endianness.NATIVE)
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=data_buff.bufsize)
    offset_buff, offset_dtype = buffers['offsets']
    offsets = buffer_to_ndarray(offset_buff, offset_dtype, offset=col.offset, length=col.size() + 1)
    null_pos = None
    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert buffers['validity'], 'Validity buffers cannot be empty for masks'
        valid_buff, valid_dtype = buffers['validity']
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, offset=col.offset, length=col.size())
        if sentinel_val == 0:
            null_pos = ~null_pos
    str_list: list[None | float | str] = [None] * col.size()
    for i in range(col.size()):
        if null_pos is not None and null_pos[i]:
            str_list[i] = np.nan
            continue
        units = data[offsets[i]:offsets[i + 1]]
        str_bytes = bytes(units)
        string = str_bytes.decode(encoding='utf-8')
        str_list[i] = string
    return (np.asarray(str_list, dtype='object'), buffers)