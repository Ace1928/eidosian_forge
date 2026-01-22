import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
@unittest.skipUnless(numpy_version < (1, 24), 'Needs NumPy < 1.24')
@TestCase.run_test_in_subprocess
def test_np_MachAr_deprecation_np122(self):
    msg = '.*`np.MachAr` is deprecated \\(NumPy 1.22\\)'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('ignore')
        warnings.filterwarnings('always', message=msg, category=NumbaDeprecationWarning)
        f = njit(lambda: np.MachAr().eps)
        f()
    self.assertEqual(len(w), 1)
    self.assertIn('`np.MachAr` is deprecated', str(w[0]))