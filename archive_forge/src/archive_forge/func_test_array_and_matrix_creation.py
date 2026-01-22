from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_array_and_matrix_creation():
    """Test of custom array creation"""
    arr = unumpy.uarray([1, 2], [0.1, 0.2])
    assert arr[1].nominal_value == 2
    assert arr[1].std_dev == 0.2
    mat = unumpy.umatrix([1, 2], [0.1, 0.2])
    assert mat[0, 1].nominal_value == 2
    assert mat[0, 1].std_dev == 0.2