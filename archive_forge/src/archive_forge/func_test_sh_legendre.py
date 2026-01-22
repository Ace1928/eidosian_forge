import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_sh_legendre(self):
    psub = np.poly1d([2, -1])
    Ps0 = orth.sh_legendre(0)
    Ps1 = orth.sh_legendre(1)
    Ps2 = orth.sh_legendre(2)
    Ps3 = orth.sh_legendre(3)
    Ps4 = orth.sh_legendre(4)
    Ps5 = orth.sh_legendre(5)
    pse0 = orth.legendre(0)(psub)
    pse1 = orth.legendre(1)(psub)
    pse2 = orth.legendre(2)(psub)
    pse3 = orth.legendre(3)(psub)
    pse4 = orth.legendre(4)(psub)
    pse5 = orth.legendre(5)(psub)
    assert_array_almost_equal(Ps0.c, pse0.c, 13)
    assert_array_almost_equal(Ps1.c, pse1.c, 13)
    assert_array_almost_equal(Ps2.c, pse2.c, 13)
    assert_array_almost_equal(Ps3.c, pse3.c, 13)
    assert_array_almost_equal(Ps4.c, pse4.c, 12)
    assert_array_almost_equal(Ps5.c, pse5.c, 12)