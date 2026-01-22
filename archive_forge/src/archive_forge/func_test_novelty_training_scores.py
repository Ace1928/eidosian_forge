import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_novelty_training_scores(global_dtype):
    X = iris.data.astype(global_dtype)
    clf_1 = neighbors.LocalOutlierFactor()
    clf_1.fit(X)
    scores_1 = clf_1.negative_outlier_factor_
    clf_2 = neighbors.LocalOutlierFactor(novelty=True)
    clf_2.fit(X)
    scores_2 = clf_2.negative_outlier_factor_
    assert_allclose(scores_1, scores_2)