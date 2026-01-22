from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def test_broadcast_funcs():
    """
    Test of mathematical functions that work with NumPy arrays of
    numbers with uncertainties.
    """
    x = ufloat(0.2, 0.1)
    arr = numpy.array([x, 2 * x])
    assert unumpy.cos(arr)[1] == uncertainties.umath.cos(arr[1])
    assert unumpy.arccos(arr)[1] == uncertainties.umath.acos(arr[1])
    assert not hasattr(numpy, 'acos')
    assert not hasattr(unumpy, 'acos')
    assert 'acos' not in unumpy.__all__