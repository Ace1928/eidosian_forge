import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def check_records(self, records):
    for rec in records:
        self.assertIsInstance(rec['new'], numba.parfors.parfor.Parfor)