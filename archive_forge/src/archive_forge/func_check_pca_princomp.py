import numpy as np
from numpy.testing import assert_array_almost_equal
from statsmodels.sandbox.tools import pca, pcasvd
from statsmodels.multivariate.tests.results.datamlw import (
def check_pca_princomp(pcares, princomp):
    factors, evals, evecs = pcares[1:]
    msign = (evecs / princomp.coef)[0]
    assert_array_almost_equal(msign * evecs, princomp.coef, 13)
    assert_array_almost_equal(msign * factors, princomp.factors, 13)
    assert_array_almost_equal(evals, princomp.values.ravel(), 13)