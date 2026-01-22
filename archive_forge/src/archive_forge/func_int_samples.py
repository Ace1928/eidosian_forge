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
def int_samples(self, typ=np.int64):
    for start in (0, -50, 60000, 1 << 32):
        info = np.iinfo(typ)
        if not info.min <= start <= info.max:
            continue
        n = 100
        yield range(start, start + n)
        yield range(start, start + 100 * n, 100)
        yield range(start, start + 128 * n, 128)
        yield [-1]