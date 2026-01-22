import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def do_testing_exceptions(self, pair):
    with self.assertRaises(TypingError) as e:
        view = njit(gen_view(pair[0], pair[1]))
        view(1)
    self.assertIn('Changing the dtype of a 0d array is only supported if the itemsize is unchanged', str(e.exception))