import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_two_records(self):
    """
        Testing the use of two scalar records of the same type
        """
    npval1 = self.refsample1d.copy()[0]
    npval2 = self.refsample1d.copy()[1]
    nbval1 = self.nbsample1d.copy()[0]
    nbval2 = self.nbsample1d.copy()[1]
    attrs = 'abc'
    valtypes = (types.float64, types.int32, types.complex64)
    for attr, valtyp in zip(attrs, valtypes):
        expected = getattr(npval1, attr) + getattr(npval2, attr)
        nbrecord = numpy_support.from_dtype(recordtype)
        pyfunc = globals()['get_two_records_' + attr]
        cfunc = self.get_cfunc(pyfunc, (nbrecord, nbrecord))
        got = cfunc(nbval1, nbval2)
        self.assertEqual(expected, got)