import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def deriv(f, x, *arg):
    x = np.asarray(x)
    h = 1e-10
    return (f(x + h * 1j, *arg) / h).imag