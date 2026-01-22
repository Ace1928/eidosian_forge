import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_rng_zero_and_extremes(self):
    for dt in self.itype:
        lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
        ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
        tgt = ubnd - 1
        assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
        tgt = lbnd
        assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
        tgt = (lbnd + ubnd) // 2
        assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)