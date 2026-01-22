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
@pytest.mark.parametrize('method', ['lar', 'lasso'])
@pytest.mark.parametrize('return_path', [True, False])
def test_lars_path_gram_equivalent(method, return_path):
    _assert_same_lars_path_result(linear_model.lars_path_gram(Xy=Xy, Gram=G, n_samples=n_samples, method=method, return_path=return_path), linear_model.lars_path(X, y, Gram=G, method=method, return_path=return_path))