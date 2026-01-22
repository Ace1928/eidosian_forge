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
def test_numpy_randint(self):
    cfunc = self._compile_array_dist('randint', 3)
    low, high = (1000, 10000)
    size = (30, 30)
    res = cfunc(low, high, size)
    self.assertIsInstance(res, np.ndarray)
    self.assertEqual(res.shape, size)
    self.assertIn(res.dtype, (np.dtype('int32'), np.dtype('int64')))
    self.assertTrue(np.all(res >= low))
    self.assertTrue(np.all(res < high))
    mean = (low + high) / 2
    tol = (high - low) / 20
    self.assertGreaterEqual(res.mean(), mean - tol)
    self.assertLessEqual(res.mean(), mean + tol)