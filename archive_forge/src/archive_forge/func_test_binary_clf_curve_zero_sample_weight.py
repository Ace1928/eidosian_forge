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
@pytest.mark.parametrize('curve_func', CURVE_FUNCS)
def test_binary_clf_curve_zero_sample_weight(curve_func):
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5]
    sample_weight = [1, 1, 1, 0.5, 0]
    result_1 = curve_func(y_true, y_score, sample_weight=sample_weight)
    result_2 = curve_func(y_true[:-1], y_score[:-1], sample_weight=sample_weight[:-1])
    for arr_1, arr_2 in zip(result_1, result_2):
        assert_allclose(arr_1, arr_2)