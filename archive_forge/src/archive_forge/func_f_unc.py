from __future__ import division
import uncertainties
import uncertainties.core as uncert_core
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import numbers_close, arrays_close
def f_unc(mat, *args, **kwargs):
    return mat.I + args[0] * kwargs['factor']