import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def return_different_statement(x):
    if x:
        return None
    else:
        return 1.2