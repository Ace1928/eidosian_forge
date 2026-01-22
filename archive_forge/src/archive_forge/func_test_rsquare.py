from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_rsquare(self):
    x = self.x + 0.0
    mu = x.mean(0)
    x_demean = x - mu
    std = np.std(x, 0)
    x_std = x_demean / std
    pc = PCA(self.x)
    nvar = x.shape[1]
    rsquare = np.zeros(nvar + 1)
    tss = np.sum(x_std ** 2)
    for i in range(nvar + 1):
        errors = x_std - pc.project(i, transform=False, unweight=False)
        rsquare[i] = 1.0 - np.sum(errors ** 2) / tss
    assert_allclose(rsquare, pc.rsquare)
    pc = PCA(self.x, standardize=False)
    tss = np.sum(x_demean ** 2)
    for i in range(nvar + 1):
        errors = x_demean - pc.project(i, transform=False, unweight=False)
        rsquare[i] = 1.0 - np.sum(errors ** 2) / tss
    assert_allclose(rsquare, pc.rsquare)
    pc = PCA(self.x, standardize=False, demean=False)
    tss = np.sum(x ** 2)
    for i in range(nvar + 1):
        errors = x - pc.project(i, transform=False, unweight=False)
        rsquare[i] = 1.0 - np.sum(errors ** 2) / tss
    assert_allclose(rsquare, pc.rsquare)