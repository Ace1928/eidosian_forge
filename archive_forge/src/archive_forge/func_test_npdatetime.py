import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_npdatetime(self):
    a = np.datetime64('1900', 'Y')
    b = np.datetime64('1900', 'Y')
    c = np.datetime64('1900-01-01', 'D')
    d = np.datetime64('1901', 'Y')
    self.eq(a, b)
    self.ne(a, c)
    self.ne(a, d)