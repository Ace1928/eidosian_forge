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
def for_signed_dtypes_combination(names=('dtype',), full=None):
    """Decorator for parameterized test w.r.t. the product set of signed dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    return for_dtypes_combination(_signed_dtypes, names=names, full=full)