from __future__ import division
from uncertainties import unumpy, ufloat
from uncertainties.unumpy.test_unumpy import arrays_close
def test_list_pseudo_inverse():
    """Test of the pseudo-inverse"""
    x = ufloat(1, 0.1)
    y = ufloat(2, 0.1)
    mat = unumpy.matrix([[x, x], [y, 0]])
    assert arrays_close(mat.I, unumpy.ulinalg.pinv(mat), 0.0001)
    assert arrays_close(unumpy.ulinalg.inv(mat), unumpy.ulinalg.pinv(mat, 1e-15), 0.0001)
    x = ufloat(1, 0.1)
    y = ufloat(2, 0.1)
    mat1 = unumpy.matrix([[x, y]])
    mat2 = unumpy.matrix([[x, y], [1, 3 + x], [y, 2 * x]])
    assert arrays_close(mat1.I, unumpy.ulinalg.pinv(mat1, 1e-10))
    assert arrays_close(mat2.I, unumpy.ulinalg.pinv(mat2, 1e-08))