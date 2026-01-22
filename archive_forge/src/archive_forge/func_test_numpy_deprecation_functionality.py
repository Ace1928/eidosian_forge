from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import os
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises, deprecated_call
import scipy
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
def test_numpy_deprecation_functionality():
    with deprecated_call():
        x = scipy.array([1, 2, 3], dtype=scipy.float64)
        assert x.dtype == scipy.float64
        assert x.dtype == np.float64
        x = scipy.finfo(scipy.float32)
        assert x.eps == np.finfo(np.float32).eps
        assert scipy.float64 == np.float64
        assert issubclass(np.float64, scipy.float64)