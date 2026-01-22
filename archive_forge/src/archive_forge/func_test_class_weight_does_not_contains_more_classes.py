import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
def test_class_weight_does_not_contains_more_classes():
    """Check that class_weight can contain more labels than in y.

    Non-regression test for #22413
    """
    tree = DecisionTreeClassifier(class_weight={0: 1, 1: 10, 2: 20})
    tree.fit([[0, 0, 1], [1, 0, 1], [1, 2, 0]], [0, 0, 1])