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
def isdtype(v):
    if isinstance(v, numpy.dtype):
        return True
    elif isinstance(v, str):
        return True
    elif isinstance(v, type) and issubclass(v, numpy.number):
        return True
    else:
        return False