import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def check_is_integer(v, name):
    """Raises TypingError if the value is not an integer."""
    if not isinstance(v, (int, types.Integer)):
        raise TypingError('{} must be an integer'.format(name))