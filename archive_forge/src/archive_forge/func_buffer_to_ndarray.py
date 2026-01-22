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
def buffer_to_ndarray(buffer: Buffer, dtype: tuple[DtypeKind, int, str, str], *, length: int, offset: int=0) -> np.ndarray:
    """
    Build a NumPy array from the passed buffer.

    Parameters
    ----------
    buffer : Buffer
        Buffer to build a NumPy array from.
    dtype : tuple
        Data type of the buffer conforming protocol dtypes format.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    length : int, optional
        If the buffer is a bit-mask, specifies a number of bits to read
        from the buffer. Has no effect otherwise.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The returned array doesn't own the memory. The caller of this function is
    responsible for keeping the memory owner object alive as long as
    the returned NumPy array is being used.
    """
    kind, bit_width, _, _ = dtype
    column_dtype = _NP_DTYPES.get(kind, {}).get(bit_width, None)
    if column_dtype is None:
        raise NotImplementedError(f'Conversion for {dtype} is not yet supported.')
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    if bit_width == 1:
        assert length is not None, '`length` must be specified for a bit-mask buffer.'
        pa = import_optional_dependency('pyarrow')
        arr = pa.BooleanArray.from_buffers(pa.bool_(), length, [None, pa.foreign_buffer(buffer.ptr, length)], offset=offset)
        return np.asarray(arr)
    else:
        data_pointer = ctypes.cast(buffer.ptr + offset * bit_width // 8, ctypes.POINTER(ctypes_type))
        if length > 0:
            return np.ctypeslib.as_array(data_pointer, shape=(length,))
        return np.array([], dtype=ctypes_type)