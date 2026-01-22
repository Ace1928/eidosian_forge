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
@jit(nopython=True)
def numba_rands(n_to_return, choice_array):
    np.random.seed(1337)
    out = np.empty((n_to_return, 2), np.int32)
    for i in range(n_to_return):
        out[i] = np.random.choice(choice_array, 2, False)
    return out