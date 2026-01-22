import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_spg_optim(self, reset_randomstate):
    dm = 100
    ind = np.arange(dm)
    indmat = np.abs(ind[:, None] - ind[None, :])
    M = 0.8 ** indmat

    def obj(x):
        return np.dot(x, np.dot(M, x))

    def grad(x):
        return 2 * np.dot(M, x)

    def project(x):
        return x
    x = np.random.normal(size=dm)
    rslt = _spg_optim(obj, grad, x, project)
    xnew = rslt.params
    assert rslt.Converged is True
    assert_almost_equal(obj(xnew), 0, decimal=3)