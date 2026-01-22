import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def gen_optional_and_type_unification_error():
    i = 0
    yield 1j
    while True:
        i = (yield i)