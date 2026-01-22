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
def test_numpy_standard_t(self):
    r = self._follow_cpython(get_np_state_ptr())
    standard_t = jit_unary('np.random.standard_t')
    avg = np.mean([standard_t(5) for i in range(5000)])
    self.assertLess(abs(avg), 0.5)