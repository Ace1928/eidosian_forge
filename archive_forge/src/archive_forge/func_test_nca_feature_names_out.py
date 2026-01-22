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
def test_nca_feature_names_out():
    """Check `get_feature_names_out` for `NeighborhoodComponentsAnalysis`."""
    X = iris_data
    y = iris_target
    est = NeighborhoodComponentsAnalysis().fit(X, y)
    names_out = est.get_feature_names_out()
    class_name_lower = est.__class__.__name__.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(est.components_.shape[1])], dtype=object)
    assert_array_equal(names_out, expected_names_out)