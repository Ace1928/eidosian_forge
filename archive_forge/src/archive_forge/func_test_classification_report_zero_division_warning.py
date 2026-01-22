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
@pytest.mark.parametrize('zero_division', ['warn', 0, 1, np.nan])
def test_classification_report_zero_division_warning(zero_division):
    y_true, y_pred = (['a', 'b', 'c'], ['a', 'b', 'd'])
    with warnings.catch_warnings(record=True) as record:
        classification_report(y_true, y_pred, zero_division=zero_division, output_dict=True)
        if zero_division == 'warn':
            assert len(record) > 1
            for item in record:
                msg = 'Use `zero_division` parameter to control this behavior.'
                assert msg in str(item.message)
        else:
            assert not record