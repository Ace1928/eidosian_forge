from __future__ import annotations
import types
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import NumbaUtilError
def jit_user_function(func: Callable) -> Callable:
    """
    If user function is not jitted already, mark the user's function
    as jitable.

    Parameters
    ----------
    func : function
        user defined function

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency('numba')
    if numba.extending.is_jitted(func):
        numba_func = func
    elif getattr(np, func.__name__, False) is func or isinstance(func, types.BuiltinFunctionType):
        numba_func = func
    else:
        numba_func = numba.extending.register_jitable(func)
    return numba_func