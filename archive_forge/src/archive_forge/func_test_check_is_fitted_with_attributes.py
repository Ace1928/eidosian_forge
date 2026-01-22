import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
@pytest.mark.parametrize('wrap', [itemgetter(0), list, tuple], ids=['single', 'list', 'tuple'])
def test_check_is_fitted_with_attributes(wrap):
    ard = ARDRegression()
    with pytest.raises(NotFittedError, match='is not fitted yet'):
        check_is_fitted(ard, wrap(['coef_']))
    ard.fit(*make_blobs())
    check_is_fitted(ard, wrap(['coef_']))
    with pytest.raises(NotFittedError, match='is not fitted yet'):
        check_is_fitted(ard, wrap(['coef_bad_']))