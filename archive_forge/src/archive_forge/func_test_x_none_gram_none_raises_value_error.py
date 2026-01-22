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
def test_x_none_gram_none_raises_value_error():
    Xy = np.dot(X.T, y)
    with pytest.raises(ValueError, match='X and Gram cannot both be unspecified'):
        linear_model.lars_path(None, y, Gram=None, Xy=Xy)