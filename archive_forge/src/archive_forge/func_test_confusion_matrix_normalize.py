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
@pytest.mark.parametrize('normalize, cm_dtype, expected_results', [('true', 'f', 0.333333333), ('pred', 'f', 0.333333333), ('all', 'f', 0.1111111111), (None, 'i', 2)])
def test_confusion_matrix_normalize(normalize, cm_dtype, expected_results):
    y_test = [0, 1, 2] * 6
    y_pred = list(chain(*permutations([0, 1, 2])))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    assert_allclose(cm, expected_results)
    assert cm.dtype.kind == cm_dtype