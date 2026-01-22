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
def test_digamma_boundary():
    x = -np.logspace(300, -30, 100)
    y = np.array([-6.1, -5.9, 5.9, 6.1])
    x, y = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()