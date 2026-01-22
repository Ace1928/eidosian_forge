import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
def test_sparse_coef():
    clf = ElasticNet()
    clf.coef_ = [1, 2, 3]
    assert sp.issparse(clf.sparse_coef_)
    assert clf.sparse_coef_.toarray().tolist()[0] == clf.coef_