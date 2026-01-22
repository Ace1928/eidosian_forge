import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def check_datetime_types(self, letter, nb_class):

    def check(dtype, numba_type, code):
        tp = numpy_support.from_dtype(dtype)
        self.assertEqual(tp, numba_type)
        self.assertEqual(tp.unit_code, code)
        self.assertEqual(numpy_support.as_dtype(numba_type), dtype)
        self.assertEqual(numpy_support.as_dtype(tp), dtype)
    check(np.dtype(letter), nb_class(''), 14)