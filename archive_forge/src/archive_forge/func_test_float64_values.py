import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_float64_values(self):
    for tp in [float, np.float64]:
        self.ne(tp(1.0 + DBL_EPSILON), tp(1.0))