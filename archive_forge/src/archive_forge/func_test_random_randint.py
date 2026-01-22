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
def test_random_randint(self):
    for tp, max_width in [(types.int64, 2 ** 63), (types.int32, 2 ** 31)]:
        cf = njit((tp, tp))(random_randint)
        self._check_randint(cf, get_py_state_ptr(), max_width)