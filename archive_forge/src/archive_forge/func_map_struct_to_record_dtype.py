from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
def map_struct_to_record_dtype(cffi_type):
    """Convert a cffi type into a NumPy Record dtype
    """
    fields = {'names': [], 'formats': [], 'offsets': [], 'itemsize': ffi.sizeof(cffi_type)}
    is_aligned = True
    for k, v in cffi_type.fields:
        if v.bitshift != -1:
            msg = 'field {!r} has bitshift, this is not supported'
            raise ValueError(msg.format(k))
        if v.flags != 0:
            msg = 'field {!r} has flags, this is not supported'
            raise ValueError(msg.format(k))
        if v.bitsize != -1:
            msg = 'field {!r} has bitsize, this is not supported'
            raise ValueError(msg.format(k))
        dtype = numpy_support.as_dtype(map_type(v.type, use_record_dtype=True))
        fields['names'].append(k)
        fields['formats'].append(dtype)
        fields['offsets'].append(v.offset)
        is_aligned &= v.offset % dtype.alignment == 0
    return numpy_support.from_dtype(np.dtype(fields, align=is_aligned))