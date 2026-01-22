import numpy as np
import numpy.polynomial.polyutils as pu
from numpy.testing import (
def test_mapdomain(self):
    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = dom2
    res = pu.mapdomain(dom1, dom1, dom2)
    assert_almost_equal(res, tgt)
    dom1 = [0 - 1j, 2 + 1j]
    dom2 = [-2, 2]
    tgt = dom2
    x = dom1
    res = pu.mapdomain(x, dom1, dom2)
    assert_almost_equal(res, tgt)
    dom1 = [0, 4]
    dom2 = [1, 3]
    tgt = np.array([dom2, dom2])
    x = np.array([dom1, dom1])
    res = pu.mapdomain(x, dom1, dom2)
    assert_almost_equal(res, tgt)

    class MyNDArray(np.ndarray):
        pass
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.array([dom1, dom1]).view(MyNDArray)
    res = pu.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, MyNDArray))