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
@pytest.mark.parametrize('min_categories, error_msg', [([[3, 2], [2, 4]], "'min_categories' should have shape")])
def test_categoricalnb_min_categories_errors(min_categories, error_msg):
    X = np.array([[0, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, 2, 2])
    clf = CategoricalNB(alpha=1, fit_prior=False, min_categories=min_categories)
    with pytest.raises(ValueError, match=error_msg):
        clf.fit(X, y)