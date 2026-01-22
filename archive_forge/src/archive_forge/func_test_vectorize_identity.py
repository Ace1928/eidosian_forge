import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
def test_vectorize_identity(self):
    from numba.tests.npyufunc.ufuncbuilding_usecases import add
    sig = 'int32(int32, int32)'
    for identity in self._supported_identities:
        ufunc = vectorize([sig], identity=identity)(add)
        expected = None if identity == 'reorderable' else identity
        self.assertEqual(ufunc.identity, expected)
    ufunc = vectorize([sig])(add)
    self.assertIs(ufunc.identity, None)
    with self.assertRaises(ValueError):
        vectorize([sig], identity='none')(add)
    with self.assertRaises(ValueError):
        vectorize([sig], identity=2)(add)