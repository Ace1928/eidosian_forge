import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_lr_cv_scores_without_enabling_metadata_routing():
    """Test that `sample_weight` is passed correctly to the scorer in
    `LogisticRegressionCV.fit` and `LogisticRegressionCV.score` even
    when `enable_metadata_routing=False`
    """
    rng = np.random.RandomState(10)
    X, y = make_classification(n_samples=10, random_state=rng)
    X_t, y_t = make_classification(n_samples=10, random_state=rng)
    sample_weight = np.ones(len(y))
    sample_weight[:len(y) // 2] = 2
    kwargs = {'sample_weight': sample_weight}
    with config_context(enable_metadata_routing=False):
        scorer1 = get_scorer('accuracy')
        lr_cv1 = LogisticRegressionCV(scoring=scorer1)
        lr_cv1.fit(X, y, **kwargs)
        score_1 = lr_cv1.score(X_t, y_t, **kwargs)
    with config_context(enable_metadata_routing=True):
        scorer2 = get_scorer('accuracy')
        scorer2.set_score_request(sample_weight=True)
        lr_cv2 = LogisticRegressionCV(scoring=scorer2)
        lr_cv2.fit(X, y, **kwargs)
        score_2 = lr_cv2.score(X_t, y_t, **kwargs)
    assert_allclose(lr_cv1.scores_[1], lr_cv2.scores_[1])
    assert_allclose(score_1, score_2)