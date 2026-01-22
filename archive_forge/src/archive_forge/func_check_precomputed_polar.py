import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy.linalg import polar, eigh
def check_precomputed_polar(a, side, expected_u, expected_p):
    u, p = polar(a, side=side)
    assert_allclose(u, expected_u, atol=1e-15)
    assert_allclose(p, expected_p, atol=1e-15)