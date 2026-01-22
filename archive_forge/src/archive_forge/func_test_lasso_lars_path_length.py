import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_lasso_lars_path_length():
    lasso = linear_model.LassoLars()
    lasso.fit(X, y)
    lasso2 = linear_model.LassoLars(alpha=lasso.alphas_[2])
    lasso2.fit(X, y)
    assert_array_almost_equal(lasso.alphas_[:3], lasso2.alphas_)
    assert np.all(np.diff(lasso.alphas_) < 0)