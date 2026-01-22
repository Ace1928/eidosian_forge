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
def test_independent_generators(self):
    N = 10
    random_seed(1)
    py_numbers = [random_random() for i in range(N)]
    numpy_seed(2)
    np_numbers = [numpy_random() for i in range(N)]
    random_seed(1)
    numpy_seed(2)
    pairs = [(random_random(), numpy_random()) for i in range(N)]
    self.assertPreciseEqual([p[0] for p in pairs], py_numbers)
    self.assertPreciseEqual([p[1] for p in pairs], np_numbers)