import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
def test_sh_chebyu(self):
    psub = np.poly1d([2, -1])
    Us0 = orth.sh_chebyu(0)
    Us1 = orth.sh_chebyu(1)
    Us2 = orth.sh_chebyu(2)
    Us3 = orth.sh_chebyu(3)
    Us4 = orth.sh_chebyu(4)
    Us5 = orth.sh_chebyu(5)
    use0 = orth.chebyu(0)(psub)
    use1 = orth.chebyu(1)(psub)
    use2 = orth.chebyu(2)(psub)
    use3 = orth.chebyu(3)(psub)
    use4 = orth.chebyu(4)(psub)
    use5 = orth.chebyu(5)(psub)
    assert_array_almost_equal(Us0.c, use0.c, 13)
    assert_array_almost_equal(Us1.c, use1.c, 13)
    assert_array_almost_equal(Us2.c, use2.c, 13)
    assert_array_almost_equal(Us3.c, use3.c, 13)
    assert_array_almost_equal(Us4.c, use4.c, 12)
    assert_array_almost_equal(Us5.c, use5.c, 11)