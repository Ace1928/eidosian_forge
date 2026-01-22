import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_vectorize_explicit_signature(self):

    def _check_explicit_signature(sig):
        f = vectorize([sig], nopython=True)(mul_usecase)
        self.assertPreciseEqual(f(TD(2), 3), TD(6))
    sig = types.NPTimedelta('s')(types.NPTimedelta('s'), types.int64)
    _check_explicit_signature(sig)
    sig = "NPTimedelta('s')(NPTimedelta('s'), int64)"
    _check_explicit_signature(sig)