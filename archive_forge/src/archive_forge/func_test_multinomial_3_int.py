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
def test_multinomial_3_int(self):
    """
        Test multinomial(n, pvals, size: int)
        """
    cfunc = jit(nopython=True)(numpy_multinomial3)
    n, pvals = (1000, self.pvals)
    k = 10
    res = cfunc(n, pvals, k)
    self.assertEqual(res.shape[0], k)
    for sample in res:
        self._check_sample(n, pvals, sample)