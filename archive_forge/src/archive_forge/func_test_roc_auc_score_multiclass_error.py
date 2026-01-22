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
@pytest.mark.parametrize('msg, kwargs', [("average must be one of \\('macro', 'weighted', None\\) for multiclass problems", {'average': 'samples', 'multi_class': 'ovo'}), ("average must be one of \\('micro', 'macro', 'weighted', None\\) for multiclass problems", {'average': 'samples', 'multi_class': 'ovr'}), ("sample_weight is not supported for multiclass one-vs-one ROC AUC, 'sample_weight' must be None in this case", {'multi_class': 'ovo', 'sample_weight': []}), ("Partial AUC computation not available in multiclass setting, 'max_fpr' must be set to `None`, received `max_fpr=0.5` instead", {'multi_class': 'ovo', 'max_fpr': 0.5}), ("multi_class must be in \\('ovo', 'ovr'\\)", {})])
def test_roc_auc_score_multiclass_error(msg, kwargs):
    rng = check_random_state(404)
    y_score = rng.rand(20, 3)
    y_prob = softmax(y_score)
    y_true = rng.randint(0, 3, size=20)
    with pytest.raises(ValueError, match=msg):
        roc_auc_score(y_true, y_prob, **kwargs)