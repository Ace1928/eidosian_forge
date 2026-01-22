import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_random_integers_max_int(self):
    with suppress_warnings() as sup:
        w = sup.record(DeprecationWarning)
        actual = random.random_integers(np.iinfo('l').max, np.iinfo('l').max)
        assert_(len(w) == 1)
    desired = np.iinfo('l').max
    assert_equal(actual, desired)
    with suppress_warnings() as sup:
        w = sup.record(DeprecationWarning)
        typer = np.dtype('l').type
        actual = random.random_integers(typer(np.iinfo('l').max), typer(np.iinfo('l').max))
        assert_(len(w) == 1)
    assert_equal(actual, desired)