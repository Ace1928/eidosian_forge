import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def select_array_wrapper(inputs):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    select the appropriate input for wrapping the operation output,
    according to each input's __array_priority__.

    An index into *inputs* is returned.
    """
    max_prio = float('-inf')
    selected_index = None
    for index, ty in enumerate(inputs):
        if isinstance(ty, types.ArrayCompatible) and ty.array_priority > max_prio:
            selected_index = index
            max_prio = ty.array_priority
    assert selected_index is not None
    return selected_index