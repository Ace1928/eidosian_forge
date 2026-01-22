import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
@unittest.expectedFailure
def test_1d_dtype_str_structured_dtype(self):
    super().test_1d_dtype_str_structured_dtype()