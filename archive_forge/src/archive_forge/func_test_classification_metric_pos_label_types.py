import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('metric', [jaccard_score, f1_score, partial(fbeta_score, beta=0.5), precision_recall_fscore_support, precision_score, recall_score, brier_score_loss])
@pytest.mark.parametrize('classes', [(False, True), (0, 1), (0.0, 1.0), ('zero', 'one')])
def test_classification_metric_pos_label_types(metric, classes):
    """Check that the metric works with different types of `pos_label`.

    We can expect `pos_label` to be a bool, an integer, a float, a string.
    No error should be raised for those types.
    """
    rng = np.random.RandomState(42)
    n_samples, pos_label = (10, classes[-1])
    y_true = rng.choice(classes, size=n_samples, replace=True)
    if metric is brier_score_loss:
        y_pred = rng.uniform(size=n_samples)
    else:
        y_pred = y_true.copy()
    result = metric(y_true, y_pred, pos_label=pos_label)
    assert not np.any(np.isnan(result))