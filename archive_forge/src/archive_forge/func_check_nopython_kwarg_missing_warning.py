import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def check_nopython_kwarg_missing_warning(self, w):
    msg = "The 'nopython' keyword argument was not supplied"
    self.assertEqual(w.category, NumbaDeprecationWarning)
    self.assertIn(msg, str(w.message))