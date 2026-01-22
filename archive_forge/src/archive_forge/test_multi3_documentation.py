import random
import numpy as np
from numba import njit
from numba.core import types
import unittest

    This test is only relevant for 32-bit architectures.

    Test __multi3 implementation in _helperlib.c.
    The symbol defines a i128 multiplication.
    It is necessary for working around an issue in LLVM (see issue #969).
    The symbol does not exist in 32-bit platform, and should not be used by
    LLVM.  However, optimization passes will create i65 multiplication that
    is then lowered to __multi3.
    