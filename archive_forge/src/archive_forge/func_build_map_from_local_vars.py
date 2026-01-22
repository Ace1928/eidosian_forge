import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def build_map_from_local_vars():
    x = TestCase
    return {0: x, x: 1}