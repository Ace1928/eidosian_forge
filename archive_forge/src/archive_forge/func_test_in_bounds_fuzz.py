import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_in_bounds_fuzz(self):
    random.seed()
    for dt in self.itype[1:]:
        for ubnd in [4, 8, 16]:
            vals = self.rfunc(2, ubnd, size=2 ** 16, dtype=dt)
            assert_(vals.max() < ubnd)
            assert_(vals.min() >= 2)
    vals = self.rfunc(0, 2, size=2 ** 16, dtype=np.bool_)
    assert_(vals.max() < 2)
    assert_(vals.min() >= 0)