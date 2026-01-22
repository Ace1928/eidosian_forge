import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def naive_hypot(x, y):
    return math.sqrt(x * x + y * y)