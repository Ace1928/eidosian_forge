import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def choose_types(numba_types, ufunc_letters):
    """
        Return a list of Numba types representing *ufunc_letters*,
        except when the letter designates a datetime64 or timedelta64,
        in which case the type is taken from *numba_types*.
        """
    assert len(ufunc_letters) >= len(numba_types)
    types = [tp if letter in 'mM' else from_dtype(np.dtype(letter)) for tp, letter in zip(numba_types, ufunc_letters)]
    types += [from_dtype(np.dtype(letter)) for letter in ufunc_letters[len(numba_types):]]
    return types