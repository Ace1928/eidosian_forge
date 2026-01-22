import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
@pytest.mark.parametrize('metric', [average_precision_score, det_curve, precision_recall_curve, roc_curve])
@pytest.mark.parametrize('classes', [(False, True), (0, 1), (0.0, 1.0), ('zero', 'one')])
def test_ranking_metric_pos_label_types(metric, classes):
    """Check that the metric works with different types of `pos_label`.

    We can expect `pos_label` to be a bool, an integer, a float, a string.
    No error should be raised for those types.
    """
    rng = np.random.RandomState(42)
    n_samples, pos_label = (10, classes[-1])
    y_true = rng.choice(classes, size=n_samples, replace=True)
    y_proba = rng.rand(n_samples)
    result = metric(y_true, y_proba, pos_label=pos_label)
    if isinstance(result, float):
        assert not np.isnan(result)
    else:
        metric_1, metric_2, thresholds = result
        assert not np.isnan(metric_1).any()
        assert not np.isnan(metric_2).any()
        assert not np.isnan(thresholds).any()