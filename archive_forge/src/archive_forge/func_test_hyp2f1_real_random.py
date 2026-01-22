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
@check_version(mpmath, '0.12')
@pytest.mark.slow
def test_hyp2f1_real_random():
    npoints = 500
    dataset = np.zeros((npoints, 5), np.float64)
    np.random.seed(1234)
    dataset[:, 0] = np.random.pareto(1.5, npoints)
    dataset[:, 1] = np.random.pareto(1.5, npoints)
    dataset[:, 2] = np.random.pareto(1.5, npoints)
    dataset[:, 3] = 2 * np.random.rand(npoints) - 1
    dataset[:, 0] *= (-1) ** np.random.randint(2, npoints)
    dataset[:, 1] *= (-1) ** np.random.randint(2, npoints)
    dataset[:, 2] *= (-1) ** np.random.randint(2, npoints)
    for ds in dataset:
        if mpmath.__version__ < '0.14':
            if abs(ds[:2]).max() > abs(ds[2]):
                ds[2] = abs(ds[:2]).max()
        ds[4] = float(mpmath.hyp2f1(*tuple(ds[:4])))
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-09).check()