import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_dynamic_raise_dict(self):

    @njit
    def raise_literal_dict2():
        raise ValueError({'a': 1, 'b': 3})
    msg = '{a: 1, b: 3}'
    with self.assertRaisesRegex(ValueError, msg):
        raise_literal_dict2()