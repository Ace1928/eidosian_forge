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
def check_floats(self, typ):
    for a in self.float_samples(typ):
        self.assertEqual(a.dtype, np.dtype(typ))
        self.check_hash_values(a)