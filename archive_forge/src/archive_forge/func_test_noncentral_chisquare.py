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
def test_noncentral_chisquare(self):
    """
        Test noncentral_chisquare(df, nonc, size)
        """
    cfunc = jit(nopython=True)(numpy_noncentral_chisquare)
    sizes = (None, 10, (10,), (10, 10))
    inputs = ((0.5, 1), (1, 5), (5, 1), (100000, 1), (1, 10000))
    for (df, nonc), size in itertools.product(inputs, sizes):
        res = cfunc(df, nonc, size)
        self._check_sample(size, res)
        res = cfunc(df, np.nan, size)
        self.assertTrue(np.isnan(res).all())