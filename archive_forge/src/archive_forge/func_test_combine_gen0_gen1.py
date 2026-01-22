import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_combine_gen0_gen1(self):
    """
        Issue #1163 is observed when two generator with NRT object arguments
        is ran in sequence.  The first one does a invalid free and corrupts
        the NRT memory subsystem.  The second generator is likely to segfault
        due to corrupted NRT data structure (an invalid MemInfo).
        """
    self.test_nrt_gen0()
    self.test_nrt_gen1()