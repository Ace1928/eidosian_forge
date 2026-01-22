import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_respect_dtype_array(self, endpoint):
    for dt in self.itype:
        lbnd = 0 if dt is bool else np.iinfo(dt).min
        ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
        ubnd = ubnd - 1 if endpoint else ubnd
        dt = np.bool_ if dt is bool else dt
        sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
        assert_equal(sample.dtype, dt)
        sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint, dtype=dt)
        assert_equal(sample.dtype, dt)