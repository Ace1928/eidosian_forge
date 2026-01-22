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
@needs_blas
@skip_parfors_unsupported
def test_unsupported_prange_warns(self):
    """
        Test that prange with multiple exits issues a warning
        """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', NumbaPerformanceWarning)
        njit((types.int64,), parallel=True)(unsupported_prange)
    self.check_parfors_unsupported_prange_warning(w)