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
@pytest.mark.parametrize('Cls', GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_oob_switch(Cls):
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    est.set_params(n_estimators=110, subsample=0.5)
    est.fit(X, y)
    assert_array_equal(est.oob_improvement_[:100], np.zeros(100))
    assert_array_equal(est.oob_scores_[:100], np.zeros(100))
    assert (est.oob_improvement_[-10:] != 0.0).all()
    assert (est.oob_scores_[-10:] != 0.0).all()
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)