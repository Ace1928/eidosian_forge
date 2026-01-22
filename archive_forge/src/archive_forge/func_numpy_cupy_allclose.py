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
def numpy_cupy_allclose(rtol=1e-07, atol=0, err_msg='', verbose=True, name='xp', type_check=True, accept_error=False, sp_name=None, scipy_name=None, contiguous_check=True, *, _check_sparse_format=True):
    """Decorator that checks NumPy results and CuPy ones are close.

    Args:
         rtol(float or dict): Relative tolerance. Besides a float value, a
             dictionary that maps a dtypes to a float value can be supplied to
             adjust tolerance per dtype. If the dictionary has ``'default'``
             string as its key, its value is used as the default tolerance in
             case any dtype keys do not match.
         atol(float or dict): Absolute tolerance. Besides a float value, a
             dictionary can be supplied as ``rtol``.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyx.scipy`` module. If ``None``, no argument is given for
             the modules.
         contiguous_check(bool): If ``True``, consistency of contiguity is
             also checked.

    Decorated test fixture is required to return the arrays whose values are
    close between ``numpy`` case and ``cupy`` case.
    For example, this test case checks ``numpy.zeros`` and ``cupy.zeros``
    should return same value.

    >>> import unittest
    >>> from cupy import testing
    >>> class TestFoo(unittest.TestCase):
    ...
    ...     @testing.numpy_cupy_allclose()
    ...     def test_foo(self, xp):
    ...         # ...
    ...         # Prepare data with xp
    ...         # ...
    ...
    ...         xp_result = xp.zeros(10)
    ...         return xp_result

    .. seealso:: :func:`cupy.testing.assert_allclose`
    """
    _check_tolerance_keys(rtol, atol)
    if not type_check:
        if isinstance(rtol, dict) or isinstance(atol, dict):
            raise TypeError('When `type_check` is `False`, `rtol` and `atol` must be supplied as float.')

    def check_func(c, n):
        rtol1, atol1 = _resolve_tolerance(type_check, c, rtol, atol)
        _array.assert_allclose(c, n, rtol1, atol1, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, contiguous_check, accept_error, sp_name, scipy_name, _check_sparse_format)