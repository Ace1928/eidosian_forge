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
def test_simple_precomputed():
    _, _, coef_path_ = linear_model.lars_path(X, y, Gram=G, method='lar')
    for i, coef_ in enumerate(coef_path_.T):
        res = y - np.dot(X, coef_)
        cov = np.dot(X.T, res)
        C = np.max(abs(cov))
        eps = 0.001
        ocur = len(cov[C - eps < abs(cov)])
        if i < X.shape[1]:
            assert ocur == i + 1
        else:
            assert ocur == X.shape[1]