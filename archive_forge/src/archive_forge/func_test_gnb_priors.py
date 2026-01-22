import re
import warnings
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_gnb_priors():
    """Test whether the class prior override is properly used"""
    clf = GaussianNB(priors=np.array([0.3, 0.7])).fit(X, y)
    assert_array_almost_equal(clf.predict_proba([[-0.1, -0.1]]), np.array([[0.825303662161683, 0.174696337838317]]), 8)
    assert_array_almost_equal(clf.class_prior_, np.array([0.3, 0.7]))