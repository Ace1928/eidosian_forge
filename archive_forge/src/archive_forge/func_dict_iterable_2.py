import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
@njit
def dict_iterable_2():
    return dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])