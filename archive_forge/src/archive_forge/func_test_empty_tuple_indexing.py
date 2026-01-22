import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_empty_tuple_indexing(self, flags=enable_pyobj_flags):
    pyfunc = empty_tuple_usecase
    arraytype = types.Array(types.int32, 0, 'C')
    argtys = (arraytype,)
    cfunc = jit(argtys, **flags)(pyfunc)
    a = np.arange(1, dtype='i4').reshape(())
    self.assertPreciseEqual(pyfunc(a), cfunc(a))