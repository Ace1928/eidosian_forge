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
@pytest.mark.parametrize('Estimator, parameters', ESTIMATORS)
def test_fit_transduction(global_dtype, Estimator, parameters):
    samples = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], dtype=global_dtype)
    labels = [0, 1, -1]
    clf = Estimator(**parameters).fit(samples, labels)
    assert clf.transduction_[2] == 1