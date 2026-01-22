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
def test_loggamma_taylor():
    r = np.logspace(-16, np.log10(LOGGAMMA_TAYLOR_RADIUS), 10)
    theta = np.linspace(0, 2 * np.pi, 20)
    r, theta = np.meshgrid(r, theta)
    dz = r * np.exp(1j * theta)
    z = np.r_[1 + dz, 2 + dz].flatten()
    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()