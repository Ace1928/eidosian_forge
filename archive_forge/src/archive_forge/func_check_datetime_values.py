import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def check_datetime_values(self, func):
    """
        Test *func*() with np.datetime64 values.
        """
    self._base_check_datetime_values(func, np.datetime64, types.NPDatetime)