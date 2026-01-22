import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type
    """
    if dtype.hasobject:
        raise TypeError('Do not support dtype containing object')
    fields = []
    for name, info in dtype.fields.items():
        [elemdtype, offset] = info[:2]
        title = info[2] if len(info) == 3 else None
        ty = from_dtype(elemdtype)
        infos = {'type': ty, 'offset': offset, 'title': title}
        fields.append((name, infos))
    size = dtype.itemsize
    aligned = _is_aligned_struct(dtype)
    return types.Record(fields, size, aligned)