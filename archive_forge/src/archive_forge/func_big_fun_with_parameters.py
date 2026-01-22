import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def big_fun_with_parameters(x, y, p):
    """ Big version of sl_fun, with two parameters.

    The two differential equations represented by sl_fun are broadcast to the
    number of rows of y, rotating between the parameters p[0] and p[1].
    Here are the differential equations:

        dy[0]/dt = y[1]
        dy[1]/dt = -p[0]**2 * y[0]
        dy[2]/dt = y[3]
        dy[3]/dt = -p[1]**2 * y[2]
        dy[4]/dt = y[5]
        dy[5]/dt = -p[0]**2 * y[4]
        dy[6]/dt = y[7]
        dy[7]/dt = -p[1]**2 * y[6]
        .
        .
        .

    """
    f = np.zeros_like(y)
    f[::2] = y[1::2]
    f[1::4] = -p[0] ** 2 * y[::4]
    f[3::4] = -p[1] ** 2 * y[2::4]
    return f