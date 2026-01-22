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
@pytest.mark.parametrize('matrix_type', list(_matrices))
@pytest.mark.parametrize('cov_type_name', _all_covariance_types)
def test_covariance(self, matrix_type, cov_type_name):
    message = f'CovVia{cov_type_name} does not support {matrix_type} matrices'
    if cov_type_name not in self._cov_types[matrix_type]:
        pytest.skip(message)
    A = self._matrices[matrix_type]
    cov_type = getattr(_covariance, f'CovVia{cov_type_name}')
    preprocessing = self._covariance_preprocessing[cov_type_name]
    psd = _PSD(A, allow_singular=True)
    cov_object = cov_type(preprocessing(A))
    assert_close(cov_object.log_pdet, psd.log_pdet)
    assert_equal(cov_object.rank, psd.rank)
    assert_equal(cov_object.shape, np.asarray(A).shape)
    assert_close(cov_object.covariance, np.asarray(A))
    rng = np.random.default_rng(5292808890472453840)
    x = rng.random(size=3)
    res = cov_object.whiten(x)
    ref = x @ psd.U
    assert_close(res @ res, ref @ ref)
    if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
        assert_close(cov_object.colorize(res), x)
    x = rng.random(size=(2, 4, 3))
    res = cov_object.whiten(x)
    ref = x @ psd.U
    assert_close((res ** 2).sum(axis=-1), (ref ** 2).sum(axis=-1))
    if hasattr(cov_object, '_colorize') and 'singular' not in matrix_type:
        assert_close(cov_object.colorize(res), x)
    if hasattr(cov_object, '_colorize'):
        res = cov_object.colorize(np.eye(len(A)))
        assert_close(res.T @ res, A)