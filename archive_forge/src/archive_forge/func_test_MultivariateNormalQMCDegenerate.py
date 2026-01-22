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
def test_MultivariateNormalQMCDegenerate(self):
    seed = np.random.default_rng(16320637417581448357869821654290448620)
    engine = qmc.MultivariateNormalQMC(mean=[0.0, 0.0, 0.0], cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]], seed=seed)
    samples = engine.random(n=512)
    assert all(np.abs(samples.mean(axis=0)) < 0.01)
    assert np.abs(np.std(samples[:, 0]) - 1) < 0.01
    assert np.abs(np.std(samples[:, 1]) - 1) < 0.01
    assert np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 0.01
    for i in (0, 1, 2):
        _, pval = shapiro(samples[:, i])
        assert pval > 0.8
    cov = np.cov(samples.transpose())
    assert np.abs(cov[0, 1]) < 0.01
    assert np.abs(cov[0, 2] - 1) < 0.01
    assert all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2]) < 1e-05)