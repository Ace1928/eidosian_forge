from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_seed():

    def test_seed_sub(gkde_trail):
        n_sample = 200
        samp1 = gkde_trail.resample(n_sample)
        samp2 = gkde_trail.resample(n_sample)
        assert_raises(AssertionError, assert_allclose, samp1, samp2, atol=1e-13)
        seed = 831
        samp1 = gkde_trail.resample(n_sample, seed=seed)
        samp2 = gkde_trail.resample(n_sample, seed=seed)
        assert_allclose(samp1, samp2, atol=1e-13)
        rstate1 = np.random.RandomState(seed=138)
        samp1 = gkde_trail.resample(n_sample, seed=rstate1)
        rstate2 = np.random.RandomState(seed=138)
        samp2 = gkde_trail.resample(n_sample, seed=rstate2)
        assert_allclose(samp1, samp2, atol=1e-13)
        if hasattr(np.random, 'default_rng'):
            rng = np.random.default_rng(1234)
            gkde_trail.resample(n_sample, seed=rng)
    np.random.seed(8928678)
    n_basesample = 500
    wn = np.random.rand(n_basesample)
    xn_1d = np.random.randn(n_basesample)
    gkde_1d = stats.gaussian_kde(xn_1d)
    test_seed_sub(gkde_1d)
    gkde_1d_weighted = stats.gaussian_kde(xn_1d, weights=wn)
    test_seed_sub(gkde_1d_weighted)
    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])
    xn_2d = np.random.multivariate_normal(mean, covariance, size=n_basesample).T
    gkde_2d = stats.gaussian_kde(xn_2d)
    test_seed_sub(gkde_2d)
    gkde_2d_weighted = stats.gaussian_kde(xn_2d, weights=wn)
    test_seed_sub(gkde_2d_weighted)