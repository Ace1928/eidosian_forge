from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_correlated_values_correlation_mat():
    """
        Tests the input of correlated value.

        Test through their correlation matrix (instead of the
        covariance matrix).
        """
    x = ufloat(1, 0.1)
    y = ufloat(2, 0.3)
    z = -3 * x + y
    cov_mat = uncert_core.covariance_matrix([x, y, z])
    std_devs = numpy.sqrt(numpy.array(cov_mat).diagonal())
    corr_mat = cov_mat / std_devs / std_devs[numpy.newaxis].T
    assert (corr_mat - corr_mat.T).max() <= 1e-15
    assert (corr_mat.diagonal() - 1).max() <= 1e-15
    nominal_values = [v.nominal_value for v in (x, y, z)]
    std_devs = [v.std_dev for v in (x, y, z)]
    x2, y2, z2 = uncert_core.correlated_values_norm(list(zip(nominal_values, std_devs)), corr_mat)
    assert arrays_close(numpy.array([x]), numpy.array([x2]))
    assert arrays_close(numpy.array([y]), numpy.array([y2]))
    assert arrays_close(numpy.array([z]), numpy.array([z2]))
    assert arrays_close(numpy.array([0]), numpy.array([z2 - (-3 * x2 + y2)]))
    assert arrays_close(numpy.array(cov_mat), numpy.array(uncert_core.covariance_matrix([x2, y2, z2])))