from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_against_reference(self):
    x = data.xo / 1000.0
    pc = PCA(x, normalize=False, standardize=False)
    ref = princomp1
    assert_allclose(np.abs(pc.factors), np.abs(ref.factors))
    assert_allclose(pc.factors.dot(pc.coeff) + x.mean(0), x)
    assert_allclose(np.abs(pc.coeff), np.abs(ref.coef.T))
    assert_allclose(pc.factors.dot(pc.coeff), ref.factors.dot(ref.coef.T))
    pc = PCA(x[:20], normalize=False, standardize=False)
    mu = x[:20].mean(0)
    ref = princomp2
    assert_allclose(np.abs(pc.factors), np.abs(ref.factors))
    assert_allclose(pc.factors.dot(pc.coeff) + mu, x[:20])
    assert_allclose(np.abs(pc.coeff), np.abs(ref.coef.T))
    assert_allclose(pc.factors.dot(pc.coeff), ref.factors.dot(ref.coef.T))