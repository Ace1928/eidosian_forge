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
def test_correlated_values():
    """
        Correlated variables.
        Test through the input of the (full) covariance matrix.
        """
    u = uncert_core.ufloat(1, 0.1)
    cov = uncert_core.covariance_matrix([u])
    u2, = uncert_core.correlated_values([1], cov)
    expr = 2 * u2
    x = ufloat(1, 0.1)
    y = ufloat(2, 0.3)
    z = -3 * x + y
    covs = uncert_core.covariance_matrix([x, y, z])
    assert arrays_close(numpy.array([v.std_dev ** 2 for v in (x, y, z)]), numpy.array(covs).diagonal())
    x_new, y_new, z_new = uncert_core.correlated_values([x.nominal_value, y.nominal_value, z.nominal_value], covs, tags=['x', 'y', 'z'])
    assert arrays_close(numpy.array((x, y, z)), numpy.array((x_new, y_new, z_new)))
    assert arrays_close(numpy.array(covs), numpy.array(uncert_core.covariance_matrix([x_new, y_new, z_new])))
    assert arrays_close(numpy.array([z_new]), numpy.array([-3 * x_new + y_new]))
    u = ufloat(1, 0.05)
    v = ufloat(10, 0.1)
    sum_value = u + 2 * v
    cov_matrix = uncert_core.covariance_matrix([u, v, sum_value])
    u2, v2, sum2 = uncert_core.correlated_values([x.nominal_value for x in [u, v, sum_value]], cov_matrix)
    assert arrays_close(numpy.array([u]), numpy.array([u2]))
    assert arrays_close(numpy.array([v]), numpy.array([v2]))
    assert arrays_close(numpy.array([sum_value]), numpy.array([sum2]))
    assert arrays_close(numpy.array([0]), numpy.array([sum2 - (u2 + 2 * v2)]))
    corr_matrix = uncert_core.correlation_matrix([u, v, sum_value])
    assert numbers_close(corr_matrix[0, 0], 1)
    assert numbers_close(corr_matrix[1, 2], 2 * v.std_dev / sum_value.std_dev)
    cov = numpy.diag([1e-70, 1e-70, 10000000000.0])
    cov[0, 1] = cov[1, 0] = 9e-71
    cov[[0, 1], 2] = -3e-34
    cov[2, [0, 1]] = -3e-34
    variables = uncert_core.correlated_values([0] * 3, cov)
    assert numbers_close(1e+66 * cov[0, 0], 1e+66 * variables[0].s ** 2, tolerance=1e-05)
    assert numbers_close(1e+66 * cov[1, 1], 1e+66 * variables[1].s ** 2, tolerance=1e-05)
    cov = numpy.diag([0, 0, 10])
    nom_values = [1, 2, 3]
    variables = uncert_core.correlated_values(nom_values, cov)
    for variable, nom_value, variance in zip(variables, nom_values, cov.diagonal()):
        assert numbers_close(variable.n, nom_value)
        assert numbers_close(variable.s ** 2, variance)
    assert arrays_close(cov, numpy.array(uncert_core.covariance_matrix(variables)))