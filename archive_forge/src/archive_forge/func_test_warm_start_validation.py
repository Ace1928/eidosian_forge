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
def test_warm_start_validation():
    X, y = make_classification(n_samples=30, n_features=5, n_classes=4, n_redundant=0, n_informative=5, random_state=0)
    nca = NeighborhoodComponentsAnalysis(warm_start=True, max_iter=5)
    nca.fit(X, y)
    X_less_features, y = make_classification(n_samples=30, n_features=4, n_classes=4, n_redundant=0, n_informative=4, random_state=0)
    msg = f'The new inputs dimensionality ({X_less_features.shape[1]}) does not match the input dimensionality of the previously learned transformation ({nca.components_.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X_less_features, y)