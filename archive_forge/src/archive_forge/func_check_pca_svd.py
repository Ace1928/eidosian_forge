import numpy as np
from numpy.testing import assert_array_almost_equal
from statsmodels.sandbox.tools import pca, pcasvd
from statsmodels.multivariate.tests.results.datamlw import (
def check_pca_svd(pcares, pcasvdres):
    xreduced, factors, evals, evecs = pcares
    xred_svd, factors_svd, evals_svd, evecs_svd = pcasvdres
    assert_array_almost_equal(evals_svd, evals, 14)
    msign = (evecs / evecs_svd)[0]
    assert_array_almost_equal(msign * evecs_svd, evecs, 13)
    assert_array_almost_equal(msign * factors_svd, factors, 13)
    assert_array_almost_equal(xred_svd, xreduced, 13)