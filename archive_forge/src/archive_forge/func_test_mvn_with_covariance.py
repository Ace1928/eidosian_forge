import pickle
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from .test_continuous_basic import check_distribution_rvs
import numpy
import numpy as np
import scipy.linalg
from scipy.stats._multivariate import (_PSD,
from scipy.stats import (multivariate_normal, multivariate_hypergeom,
from scipy.stats import _covariance, Covariance
from scipy import stats
from scipy.integrate import romb, qmc_quad, tplquad
from scipy.special import multigammaln
from scipy._lib._pep440 import Version
from .common_tests import check_random_state_property
from .data._mvt import _qsimvtv
from unittest.mock import patch
@pytest.mark.parametrize('size', [None, tuple(), 1, (2, 4, 3)])
@pytest.mark.parametrize('matrix_type', list(_matrices))
@pytest.mark.parametrize('cov_type_name', _all_covariance_types)
def test_mvn_with_covariance(self, size, matrix_type, cov_type_name):
    message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
    if cov_type_name not in self._cov_types[matrix_type]:
        pytest.skip(message)
    A = self._matrices[matrix_type]
    cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
    preprocessing = self._covariance_preprocessing[cov_type_name]
    mean = [0.1, 0.2, 0.3]
    cov_object = cov_type(preprocessing(A))
    mvn = multivariate_normal
    dist0 = multivariate_normal(mean, A, allow_singular=True)
    dist1 = multivariate_normal(mean, cov_object, allow_singular=True)
    rng = np.random.default_rng(5292808890472453840)
    x = rng.multivariate_normal(mean, A, size=size)
    rng = np.random.default_rng(5292808890472453840)
    x1 = mvn.rvs(mean, cov_object, size=size, random_state=rng)
    rng = np.random.default_rng(5292808890472453840)
    x2 = mvn(mean, cov_object, seed=rng).rvs(size=size)
    if isinstance(cov_object, _covariance.CovViaPSD):
        assert_close(x1, np.squeeze(x))
        assert_close(x2, np.squeeze(x))
    else:
        assert_equal(x1.shape, x.shape)
        assert_equal(x2.shape, x.shape)
        assert_close(x2, x1)
    assert_close(mvn.pdf(x, mean, cov_object), dist0.pdf(x))
    assert_close(dist1.pdf(x), dist0.pdf(x))
    assert_close(mvn.logpdf(x, mean, cov_object), dist0.logpdf(x))
    assert_close(dist1.logpdf(x), dist0.logpdf(x))
    assert_close(mvn.entropy(mean, cov_object), dist0.entropy())
    assert_close(dist1.entropy(), dist0.entropy())