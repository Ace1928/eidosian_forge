from numba import njit
from numba.core import errors
from numba.core.extending import overload
import numpy as np
import unittest
@overload(foo)
def foo_ol(func, *args):
    nargs = len(args)
    if nargs == 1:

        def impl(func, *args):
            return func(*args)
        return impl
    elif nargs == 2:

        def impl(func, *args):
            return func(func(*args))
        return impl