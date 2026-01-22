import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def pressure_network_jacobian(flow_rates, Qtot, k):
    """Return the jacobian of the equation system F(flow_rates)
    computed by `pressure_network` with respect to
    *flow_rates*. See `pressure_network` for the detailed
    description of parameters.

    Returns
    -------
    jac : float
        *n* by *n* matrix ``df_i/dQ_i`` where ``n = len(flow_rates)``
        and *f_i* and *Q_i* are described in the doc for `pressure_network`
    """
    n = len(flow_rates)
    pdiff = np.diag(flow_rates[1:] * 2 * k[1:] - 2 * flow_rates[0] * k[0])
    jac = np.empty((n, n))
    jac[:n - 1, :n - 1] = pdiff * 0
    jac[:n - 1, n - 1] = 0
    jac[n - 1, :] = np.ones(n)
    return jac