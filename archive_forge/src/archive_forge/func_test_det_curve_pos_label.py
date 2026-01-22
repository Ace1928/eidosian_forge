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
def test_det_curve_pos_label():
    y_true = ['cancer'] * 3 + ['not cancer'] * 7
    y_pred_pos_not_cancer = np.array([0.1, 0.4, 0.6, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    y_pred_pos_cancer = 1 - y_pred_pos_not_cancer
    fpr_pos_cancer, fnr_pos_cancer, th_pos_cancer = det_curve(y_true, y_pred_pos_cancer, pos_label='cancer')
    fpr_pos_not_cancer, fnr_pos_not_cancer, th_pos_not_cancer = det_curve(y_true, y_pred_pos_not_cancer, pos_label='not cancer')
    assert th_pos_cancer[0] == pytest.approx(0.4)
    assert th_pos_not_cancer[0] == pytest.approx(0.2)
    assert_allclose(fpr_pos_cancer, fnr_pos_not_cancer[::-1])
    assert_allclose(fnr_pos_cancer, fpr_pos_not_cancer[::-1])