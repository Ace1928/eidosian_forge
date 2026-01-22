import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
def make_test_array(self, n_members):
    return np.arange(n_members, dtype=np.int64)