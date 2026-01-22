import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_MultivariateNormalQMCShapiroInvTransform(self):
    seed = np.random.default_rng(200089821034563288698994840831440331329)
    engine = qmc.MultivariateNormalQMC(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed, inv_transform=True)
    samples = engine.random(n=256)
    assert all(np.abs(samples.mean(axis=0)) < 0.01)
    assert all(np.abs(samples.std(axis=0) - 1) < 0.01)
    for i in (0, 1):
        _, pval = shapiro(samples[:, i])
        assert pval > 0.9
    cov = np.cov(samples.transpose())
    assert np.abs(cov[0, 1]) < 0.01
    engine = qmc.MultivariateNormalQMC(mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed, inv_transform=True)
    samples = engine.random(n=256)
    assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 0.01)
    assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 0.01)
    for i in (0, 1):
        _, pval = shapiro(samples[:, i])
        assert pval > 0.9
    cov = np.cov(samples.transpose())
    assert np.abs(cov[0, 1] - 0.5) < 0.01