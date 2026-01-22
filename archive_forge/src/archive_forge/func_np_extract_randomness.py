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
@jit(nopython=True, nogil=True)
def np_extract_randomness(seed, out):
    if seed != 0:
        np.random.seed(seed)
    s = 0
    for i in range(out.size):
        out[i] = np.random.randint(_randint_limit)