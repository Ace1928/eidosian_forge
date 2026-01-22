import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def compile_func(arr):
    argtys = (typeof(arr), types.intp, types.intp)
    return jit(argtys, **flags)(pyfunc)