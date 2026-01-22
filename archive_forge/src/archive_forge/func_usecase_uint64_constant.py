import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
def usecase_uint64_constant():
    return 18446744073709551615