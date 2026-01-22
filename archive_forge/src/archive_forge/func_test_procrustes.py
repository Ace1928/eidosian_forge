import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes(self):
    a, b, disparity = procrustes(self.data1, self.data2)
    assert_allclose(b, a)
    assert_almost_equal(disparity, 0.0)
    m4, m5, disp45 = procrustes(self.data4, self.data5)
    assert_equal(m4, self.data4)
    m1, m3, disp13 = procrustes(self.data1, self.data3)