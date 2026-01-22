import warnings
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import _label_propagation as label_propagation
from sklearn.utils._testing import (
@pytest.mark.parametrize('constructor_type', CONSTRUCTOR_TYPES)
def test_convergence_speed(constructor_type):
    X = _convert_container([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], constructor_type)
    y = np.array([0, 1, -1])
    mdl = label_propagation.LabelSpreading(kernel='rbf', max_iter=5000)
    mdl.fit(X, y)
    assert mdl.n_iter_ < 10
    assert_array_equal(mdl.predict(X), [0, 1, 1])