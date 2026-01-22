import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api
def test_kmeans_and_kmeans2_random_seed(self):
    seed_list = [1234, np.random.RandomState(1234), np.random.default_rng(1234)]
    for seed in seed_list:
        res1, _ = kmeans(TESTDATA_2D, 2, seed=seed)
        res2, _ = kmeans(TESTDATA_2D, 2, seed=seed)
        assert_allclose(res1, res1)
        for minit in ['random', 'points', '++']:
            res1, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
            res2, _ = kmeans2(TESTDATA_2D, 2, minit=minit, seed=seed)
            assert_allclose(res1, res1)