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
@check_version(mpmath, '1.1.0')
def test_hyperu_around_0():
    dataset = []
    for n in np.arange(-5, 5):
        for b in np.linspace(-5, 5, 20):
            a = -n
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
            a = -n + b - 1
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    for a in [-10.5, -1.5, -0.5, 0, 0.5, 1, 10]:
        for b in [-1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    dataset = np.array(dataset)
    FuncData(sc.hyperu, dataset, (0, 1, 2), 3, rtol=1e-15, atol=5e-13).check()