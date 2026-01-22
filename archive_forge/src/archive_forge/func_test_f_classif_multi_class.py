import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_f_classif_multi_class():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=3, n_redundant=2, n_repeated=0, n_classes=8, n_clusters_per_class=1, flip_y=0.0, class_sep=10, shuffle=False, random_state=0)
    F, pv = f_classif(X, y)
    assert (F > 0).all()
    assert (pv > 0).all()
    assert (pv < 1).all()
    assert (pv[:5] < 0.05).all()
    assert (pv[5:] > 0.0001).all()