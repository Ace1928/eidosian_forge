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
def test_sinpi_zeros():
    eps = np.finfo(float).eps
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    zeros = np.arange(-100, 100, 1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,) * zeros.size)).flatten()
    dataset = np.asarray([(z0, complex(mpmath.sinpi(z0))) for z0 in z])
    FuncData(_sinpi, dataset, 0, 1, rtol=2 * eps).check()