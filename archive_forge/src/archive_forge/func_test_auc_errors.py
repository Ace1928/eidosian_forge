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
def test_auc_errors():
    with pytest.raises(ValueError):
        auc([0.0, 0.5, 1.0], [0.1, 0.2])
    with pytest.raises(ValueError):
        auc([0.0], [0.1])
    x = [2, 1, 3, 4]
    y = [5, 6, 7, 8]
    error_message = 'x is neither increasing nor decreasing : {}'.format(np.array(x))
    with pytest.raises(ValueError, match=re.escape(error_message)):
        auc(x, y)