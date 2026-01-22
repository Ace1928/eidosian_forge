import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
def check_svml_presence(self, func, pattern):
    asm = func.library.get_asm_str()
    self.assertIn(pattern, asm)