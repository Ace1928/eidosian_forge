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
def float_samples(self, typ):
    info = np.finfo(typ)
    for start in (0, 10, info.max ** 0.5, info.max / 1000.0):
        n = 100
        min_step = max(info.tiny, start * info.resolution)
        for step in (1.2, min_step ** 0.5, min_step):
            if step < min_step:
                continue
            a = np.linspace(start, start + n * step, n)
            a = a.astype(typ)
            yield a
            yield (-a)
            yield (a + a.mean())
    a = [0.0, 0.5, -0.0, -1.0, float('inf'), -float('inf')]
    if utils.PYVERSION < (3, 10):
        a.append(float('nan'))
    yield typ(a)