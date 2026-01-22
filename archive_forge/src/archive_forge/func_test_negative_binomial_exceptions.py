import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_negative_binomial_exceptions(self):
    with suppress_warnings() as sup:
        sup.record(RuntimeWarning)
        assert_raises(ValueError, random.negative_binomial, 100, np.nan)
        assert_raises(ValueError, random.negative_binomial, 100, [np.nan] * 10)