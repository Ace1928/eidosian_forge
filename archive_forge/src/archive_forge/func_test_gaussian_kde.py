import itertools
import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.metrics import DistanceMetric
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._ball_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.neighbors._kd_tree import (
from sklearn.utils import check_random_state
@pytest.mark.parametrize('Cls', [KDTree, BallTree])
def test_gaussian_kde(Cls, n_samples=1000):
    from scipy.stats import gaussian_kde
    rng = check_random_state(0)
    x_in = rng.normal(0, 1, n_samples)
    x_out = np.linspace(-5, 5, 30)
    for h in [0.01, 0.1, 1]:
        tree = Cls(x_in[:, None])
        gkde = gaussian_kde(x_in, bw_method=h / np.std(x_in))
        dens_tree = tree.kernel_density(x_out[:, None], h) / n_samples
        dens_gkde = gkde.evaluate(x_out)
        assert_array_almost_equal(dens_tree, dens_gkde, decimal=3)