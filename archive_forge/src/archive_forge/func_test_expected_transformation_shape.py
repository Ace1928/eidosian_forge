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
def test_expected_transformation_shape():
    """Test that the transformation has the expected shape."""
    X = iris_data
    y = iris_target

    class TransformationStorer:

        def __init__(self, X, y):
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            self.X, y = self.fake_nca._validate_data(X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            """Stores the last value of the transformation taken as input by
            the optimizer"""
            self.transformation = transformation
    transformation_storer = TransformationStorer(X, y)
    cb = transformation_storer.callback
    nca = NeighborhoodComponentsAnalysis(max_iter=5, callback=cb)
    nca.fit(X, y)
    assert transformation_storer.transformation.size == X.shape[1] ** 2