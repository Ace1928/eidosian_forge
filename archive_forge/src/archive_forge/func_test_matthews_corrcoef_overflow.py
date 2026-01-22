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
@pytest.mark.parametrize('n_points', [100, 10000])
def test_matthews_corrcoef_overflow(n_points):
    rng = np.random.RandomState(20170906)

    def mcc_safe(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        true_pos = conf_matrix[1, 1]
        false_pos = conf_matrix[1, 0]
        false_neg = conf_matrix[0, 1]
        n_points = len(y_true)
        pos_rate = (true_pos + false_neg) / n_points
        activity = (true_pos + false_pos) / n_points
        mcc_numerator = true_pos / n_points - pos_rate * activity
        mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
        return mcc_numerator / np.sqrt(mcc_denominator)

    def random_ys(n_points):
        x_true = rng.random_sample(n_points)
        x_pred = x_true + 0.2 * (rng.random_sample(n_points) - 0.5)
        y_true = x_true > 0.5
        y_pred = x_pred > 0.5
        return (y_true, y_pred)
    arr = np.repeat([0.0, 1.0], n_points)
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)
    arr = np.repeat([0.0, 1.0, 2.0], n_points)
    assert_almost_equal(matthews_corrcoef(arr, arr), 1.0)
    y_true, y_pred = random_ys(n_points)
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), mcc_safe(y_true, y_pred))