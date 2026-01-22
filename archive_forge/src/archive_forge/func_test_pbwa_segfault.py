import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
def test_pbwa_segfault():
    w = 1.0227656721131686
    wp = -0.4888705337234619
    assert_allclose(sc.pbwa(0, 0), (w, wp), rtol=1e-13, atol=0)