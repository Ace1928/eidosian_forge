import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_spence_circle():

    def spence(z):
        return complex(mpmath.polylog(2, 1 - z))
    r = np.linspace(0.5, 1.5)
    theta = np.linspace(0, 2 * pi)
    z = (1 + np.outer(r, np.exp(1j * theta))).flatten()
    dataset = np.asarray([(z0, spence(z0)) for z0 in z])
    FuncData(sc.spence, dataset, 0, 1, rtol=1e-14).check()