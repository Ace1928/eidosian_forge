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
def test_dirichlet(self):
    """
        Test dirichlet(alpha, size=None)
        """
    cfunc = jit(nopython=True)(numpy_dirichlet)
    sizes = (None, (10,), (10, 10))
    alphas = (self.alpha, tuple(self.alpha), np.array([1, 1, 10000, 1], dtype=np.float64), np.array([1, 1, 1.5, 1], dtype=np.float64))
    for alpha, size in itertools.product(alphas, sizes):
        res = cfunc(alpha, size)
        self._check_sample(alpha, size, res)