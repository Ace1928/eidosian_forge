import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
def test_integer_repeat(int_func):
    random.seed(123456789)
    fname, args, sha256 = int_func
    f = getattr(random, fname)
    val = f(*args, size=1000000)
    if sys.byteorder != 'little':
        val = val.byteswap()
    res = hashlib.sha256(val.view(np.int8)).hexdigest()
    assert_(res == sha256)