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
def test_numpy_rand(self):
    cfunc = jit(nopython=True)(numpy_check_rand)
    expected, got = cfunc(42, 2, 3)
    self.assertEqual(got.shape, (2, 3))
    self.assertPreciseEqual(expected, got)