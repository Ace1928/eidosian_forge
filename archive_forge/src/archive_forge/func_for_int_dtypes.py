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
def for_int_dtypes(name='dtype', no_bool=False):
    """Decorator that checks the fixture with integer and optionally bool dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    if no_bool:
        return for_dtypes(_int_dtypes, name=name)
    else:
        return for_dtypes(_int_bool_dtypes, name=name)