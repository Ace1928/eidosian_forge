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
def test_numpy_uniform_kwargs(self):
    self._check_any_distrib_kwargs(jit_with_kwargs('np.random.uniform', ['low', 'high']), get_np_state_ptr(), 'uniform', paramlist=[{'low': 1.5, 'high': 1000000.0}, {'low': -2.5, 'high': 1000.0}, {'low': 1.5, 'high': -2.5}])