import unittest
import os
import sys
import subprocess
from collections import defaultdict
from textwrap import dedent
import numpy as np
from numba import jit, config, typed, typeof
from numba.core import types, utils
import unittest
from numba.tests.support import (TestCase, skip_unless_py10_or_later,
from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing
@skip_unless_py10_or_later
def test_py310_nan_hash(self):
    x = [float('nan') for i in range(10)]
    out = set([self.cfunc(z) for z in x])
    self.assertGreater(len(out), 1)