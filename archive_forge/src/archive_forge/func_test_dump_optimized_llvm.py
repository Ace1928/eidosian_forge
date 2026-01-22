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
def test_dump_optimized_llvm(self):
    with override_config('DUMP_OPTIMIZED', True):
        out = self.compile_simple_nopython()
    self.check_debug_output(out, ['optimized_llvm'])