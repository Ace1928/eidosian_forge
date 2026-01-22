import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def make_specific(outputs, unit):
    new_outputs = []
    for out in outputs:
        if isinstance(out, types.NPTimedelta) and out.unit == '':
            new_outputs.append(types.NPTimedelta(unit))
        else:
            new_outputs.append(out)
    return new_outputs