import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_chebyc(self):
    C0 = orth.chebyc(0)
    C1 = orth.chebyc(1)
    with np.errstate(all='ignore'):
        C2 = orth.chebyc(2)
        C3 = orth.chebyc(3)
        C4 = orth.chebyc(4)
        C5 = orth.chebyc(5)
    assert_array_almost_equal(C0.c, [2], 13)
    assert_array_almost_equal(C1.c, [1, 0], 13)
    assert_array_almost_equal(C2.c, [1, 0, -2], 13)
    assert_array_almost_equal(C3.c, [1, 0, -3, 0], 13)
    assert_array_almost_equal(C4.c, [1, 0, -4, 0, 2], 13)
    assert_array_almost_equal(C5.c, [1, 0, -5, 0, 5, 0], 13)