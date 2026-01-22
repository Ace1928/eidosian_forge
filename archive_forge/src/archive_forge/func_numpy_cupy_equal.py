import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def numpy_cupy_equal(name='xp', sp_name=None, scipy_name=None):
    """Decorator that checks NumPy results are equal to CuPy ones.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same results
    even if ``xp`` is ``numpy`` or ``cupy``.
    """

    def decorator(impl):

        @_wraps_partial_xp(impl, name, sp_name, scipy_name)
        def test_func(*args, **kw):
            cupy_result, cupy_error, numpy_result, numpy_error = _call_func_numpy_cupy(impl, args, kw, name, sp_name, scipy_name)
            if cupy_error or numpy_error:
                _check_cupy_numpy_error(cupy_error, numpy_error, accept_error=False)
                return
            if cupy_result != numpy_result:
                message = 'Results are not equal:\ncupy: %s\nnumpy: %s' % (str(cupy_result), str(numpy_result))
                raise AssertionError(message)
        return test_func
    return decorator