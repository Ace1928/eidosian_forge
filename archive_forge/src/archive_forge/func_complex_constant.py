import numpy as np
import unittest
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase
def complex_constant(n):
    tmp = n + 4
    return tmp + 3j