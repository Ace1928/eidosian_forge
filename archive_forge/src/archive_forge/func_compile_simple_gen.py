import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
def compile_simple_gen(self):
    with captured_stdout() as out:
        cfunc = njit((types.int64, types.int64))(simple_gen)
        self.assertPreciseEqual(list(cfunc(2, 5)), [2, 5])
    return out.getvalue()