import scipy.special as sc
import numpy as np
from numpy.testing import assert_equal, assert_allclose
def test_riemann_zeta_avoid_overflow():
    s = -260.00000000001
    desired = -5.696630784440268e+297
    assert_allclose(sc.zeta(s), desired, atol=0, rtol=5e-14)