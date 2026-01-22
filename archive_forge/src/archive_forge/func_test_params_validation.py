import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
def test_params_validation():
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    NCA = NeighborhoodComponentsAnalysis
    rng = np.random.RandomState(42)
    init = rng.rand(5, 3)
    msg = f'The output dimensionality ({init.shape[0]}) of the given linear transformation `init` cannot be greater than its input dimensionality ({init.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(init=init).fit(X, y)
    n_components = 10
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) cannot be greater than the given data dimensionality ({X.shape[1]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(n_components=n_components).fit(X, y)