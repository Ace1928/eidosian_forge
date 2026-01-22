import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
def test_out_size_mismatch(self):
    out = np.zeros(10)
    assert_raises(ValueError, random.standard_gamma, 10.0, size=20, out=out)
    assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1), out=out)