import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb import _safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR
from sklearn.utils import check_random_state, tosequence
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_check_inputs_predict_stages(csc_container):
    x, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    x_sparse_csc = csc_container(x)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(x, y)
    score = np.zeros(y.shape).reshape(-1, 1)
    err_msg = 'When X is a sparse matrix, a CSR format is expected'
    with pytest.raises(ValueError, match=err_msg):
        predict_stages(clf.estimators_, x_sparse_csc, clf.learning_rate, score)
    x_fortran = np.asfortranarray(x)
    with pytest.raises(ValueError, match='X should be C-ordered np.ndarray'):
        predict_stages(clf.estimators_, x_fortran, clf.learning_rate, score)