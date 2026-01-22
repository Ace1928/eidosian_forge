import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_number_types(self, tp_factory):
    values = [0, 1, -1, 100003, 10000000000007, -100003, -10000000000007, 1.5, -3.5]
    for tp_name in real_np_types():
        np_type = getattr(np, tp_name)
        tp = tp_factory(tp_name)
        self.check_type_converter(tp, np_type, values)
    values.append(1.5 + 3j)
    for tp_name in complex_np_types():
        np_type = getattr(np, tp_name)
        tp = tp_factory(tp_name)
        self.check_type_converter(tp, np_type, values)