import numpy as np
import numpy.core as nx
import numpy.lib.ufunclike as ufl
from numpy.testing import (
def test_isneginf(self):
    a = nx.array([nx.inf, -nx.inf, nx.nan, 0.0, 3.0, -3.0])
    out = nx.zeros(a.shape, bool)
    tgt = nx.array([False, True, False, False, False, False])
    res = ufl.isneginf(a)
    assert_equal(res, tgt)
    res = ufl.isneginf(a, out)
    assert_equal(res, tgt)
    assert_equal(out, tgt)
    a = a.astype(np.complex_)
    with assert_raises(TypeError):
        ufl.isneginf(a)