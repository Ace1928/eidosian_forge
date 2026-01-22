import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_gaussian_mixture_setting_best_params():
    """`GaussianMixture`'s best_parameters, `n_iter_` and `lower_bound_`
    must be set appropriately in the case of divergence.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/18216
    """
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))
    means_init = np.array([[0.670637869618158, 0.21038256107384043, 0.12892629765485303], [0.09394051075844147, 0.5759464955561779, 0.929296197576212], [0.5033230372781258, 0.9569852381759425, 0.08654043447295741], [0.18578301420435747, 0.5531158970919143, 0.19388943970532435], [0.4548589928173794, 0.35182513658825276, 0.568146063202464], [0.609279894978321, 0.7929063819678847, 0.9620097270828052]])
    precisions_init = np.array([999999.999604483, 999999.9990869573, 553.7603944542167, 204.78596008931834, 15.867423501783637, 85.4595728389735])
    weights_init = [0.03333333333333341, 0.03333333333333341, 0.06666666666666674, 0.06666666666666674, 0.7000000000000001, 0.10000000000000007]
    gmm = GaussianMixture(covariance_type='spherical', reg_covar=0, means_init=means_init, weights_init=weights_init, random_state=rnd, n_components=len(weights_init), precisions_init=precisions_init, max_iter=1)
    gmm.fit(X)
    assert not gmm.converged_
    for attr in ['weights_', 'means_', 'covariances_', 'precisions_cholesky_', 'n_iter_', 'lower_bound_']:
        assert hasattr(gmm, attr)