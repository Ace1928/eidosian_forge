import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def test_dirichlet_exceptions(self):
    cfunc = jit(nopython=True)(numpy_dirichlet)
    alpha = tuple((0, 1, 1))
    with self.assertRaises(ValueError) as raises:
        cfunc(alpha, 1)
    self.assertIn('dirichlet: alpha must be > 0.0', str(raises.exception))
    alpha = self.alpha
    sizes = (True, 3j, 1.5, (1.5, 1), (3j, 1), (3j, 3j), (np.int8(3), np.int64(7)))
    for size in sizes:
        with self.assertRaises(TypingError) as raises:
            cfunc(alpha, size)
        self.assertIn('np.random.dirichlet(): size should be int or tuple of ints or None, got', str(raises.exception))