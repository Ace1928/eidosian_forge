import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_poisson_exceptions(self):
    lambig = np.iinfo('l').max
    lamneg = -1
    assert_raises(ValueError, random.poisson, lamneg)
    assert_raises(ValueError, random.poisson, [lamneg] * 10)
    assert_raises(ValueError, random.poisson, lambig)
    assert_raises(ValueError, random.poisson, [lambig] * 10)
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        assert_raises(ValueError, random.poisson, np.nan)
        assert_raises(ValueError, random.poisson, [np.nan] * 10)