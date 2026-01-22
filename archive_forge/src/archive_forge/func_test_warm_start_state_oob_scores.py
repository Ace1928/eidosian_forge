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
@pytest.mark.parametrize('GradientBoosting', GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_state_oob_scores(GradientBoosting):
    """
    Check that the states of the OOB scores are cleared when used with `warm_start`.
    """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    n_estimators = 100
    estimator = GradientBoosting(n_estimators=n_estimators, max_depth=1, subsample=0.5, warm_start=True, random_state=1)
    estimator.fit(X, y)
    oob_scores, oob_score = (estimator.oob_scores_, estimator.oob_score_)
    assert len(oob_scores) == n_estimators
    assert oob_scores[-1] == pytest.approx(oob_score)
    n_more_estimators = 200
    estimator.set_params(n_estimators=n_more_estimators).fit(X, y)
    assert len(estimator.oob_scores_) == n_more_estimators
    assert_allclose(estimator.oob_scores_[:n_estimators], oob_scores)
    estimator.set_params(n_estimators=n_estimators, warm_start=False).fit(X, y)
    assert estimator.oob_scores_ is not oob_scores
    assert estimator.oob_score_ is not oob_score
    assert_allclose(estimator.oob_scores_, oob_scores)
    assert estimator.oob_score_ == pytest.approx(oob_score)
    assert oob_scores[-1] == pytest.approx(oob_score)