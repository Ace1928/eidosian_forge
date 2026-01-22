import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('missing_value', [-1, np.nan])
def test_simple_imputation_inverse_transform_exceptions(missing_value):
    X_1 = np.array([[9, missing_value, 3, -1], [4, -1, 5, 4], [6, 7, missing_value, -1], [8, 9, 0, missing_value]])
    imputer = SimpleImputer(missing_values=missing_value, strategy='mean')
    X_1_trans = imputer.fit_transform(X_1)
    with pytest.raises(ValueError, match=f"Got 'add_indicator={imputer.add_indicator}'"):
        imputer.inverse_transform(X_1_trans)