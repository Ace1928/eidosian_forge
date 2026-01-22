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
def test_copy_X_with_auto_gram():
    rng = np.random.RandomState(42)
    X = rng.rand(6, 6)
    y = rng.rand(6)
    X_before = X.copy()
    linear_model.lars_path(X, y, Gram='auto', copy_X=True, method='lasso')
    assert_allclose(X, X_before)