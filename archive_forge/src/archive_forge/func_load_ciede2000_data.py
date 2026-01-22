import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
def load_ciede2000_data():
    dtype = [('pair', int), ('1', int), ('L1', float), ('a1', float), ('b1', float), ('a1_prime', float), ('C1_prime', float), ('h1_prime', float), ('hbar_prime', float), ('G', float), ('T', float), ('SL', float), ('SC', float), ('SH', float), ('RT', float), ('dE', float), ('2', int), ('L2', float), ('a2', float), ('b2', float), ('a2_prime', float), ('C2_prime', float), ('h2_prime', float)]
    path = fetch('color/tests/ciede2000_test_data.txt')
    return np.loadtxt(path, dtype=dtype)