import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_none_index(self, flags=enable_pyobj_flags):
    pyfunc = none_index_usecase
    arraytype = types.Array(types.int32, 2, 'C')
    argtys = (arraytype,)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(100, dtype='i4').reshape(10, 10)
    self.assertPreciseEqual(pyfunc(a), cfunc(a))