from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_array_comparisons():
    """Test of array and matrix comparisons"""
    arr = unumpy.uarray([1, 2], [1, 4])
    assert numpy.all((arr == [arr[0], 4]) == [True, False])
    mat = unumpy.umatrix([1, 2], [1, 4])
    assert numpy.all((mat == [mat[0, 0], 4]) == [True, False])