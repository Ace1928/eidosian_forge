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
@pytest.mark.parametrize('multi_class, average', [('ovr', 'macro'), ('ovr', 'micro'), ('ovo', 'macro')])
def test_perfect_imperfect_chance_multiclass_roc_auc(multi_class, average):
    y_true = np.array([3, 1, 2, 0])
    y_perfect = [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.75, 0.05, 0.05, 0.15]]
    assert_almost_equal(roc_auc_score(y_true, y_perfect, multi_class=multi_class, average=average), 1.0)
    y_imperfect = [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    assert roc_auc_score(y_true, y_imperfect, multi_class=multi_class, average=average) < 1.0
    y_chance = 0.25 * np.ones((4, 4))
    assert roc_auc_score(y_true, y_chance, multi_class=multi_class, average=average) == pytest.approx(0.5)