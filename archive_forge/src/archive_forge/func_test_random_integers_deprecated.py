import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_random_integers_deprecated(self):
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max)
        assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max, np.iinfo('l').max)