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
def test_numpy_random(self):
    self._check_random_seed(numpy_seed, numpy_random)
    self._check_random_seed(numpy_seed, jit_nullary('np.random.random_sample'))
    self._check_random_seed(numpy_seed, jit_nullary('np.random.ranf'))
    self._check_random_seed(numpy_seed, jit_nullary('np.random.sample'))
    self._check_random_seed(numpy_seed, jit_nullary('np.random.rand'))