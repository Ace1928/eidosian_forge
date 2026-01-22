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
def test_dcg_ties():
    y_true = np.asarray([np.arange(5)])
    y_score = np.zeros(y_true.shape)
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    discounts = 1 / np.log2(np.arange(2, 7))
    assert dcg == pytest.approx([discounts.sum() * y_true.mean()])
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    y_score[0, 3:] = 1
    dcg = _dcg_sample_scores(y_true, y_score)
    dcg_ignore_ties = _dcg_sample_scores(y_true, y_score, ignore_ties=True)
    assert dcg_ignore_ties == pytest.approx([(discounts * y_true[:, ::-1]).sum()])
    assert dcg == pytest.approx([discounts[:2].sum() * y_true[0, 3:].mean() + discounts[2:].sum() * y_true[0, :3].mean()])