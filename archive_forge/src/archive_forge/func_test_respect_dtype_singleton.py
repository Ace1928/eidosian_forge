import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_respect_dtype_singleton(self):
    for dt in self.itype:
        lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
        ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
        sample = self.rfunc(lbnd, ubnd, dtype=dt)
        assert_equal(sample.dtype, np.dtype(dt))
    for dt in (bool, int):
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        sample = self.rfunc(lbnd, ubnd, dtype=dt)
        assert_(not hasattr(sample, 'dtype'))
        assert_equal(type(sample), dt)