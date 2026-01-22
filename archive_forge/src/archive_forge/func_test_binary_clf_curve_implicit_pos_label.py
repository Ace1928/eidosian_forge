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
def test_binary_clf_curve_implicit_pos_label(curve_func):
    msg = "y_true takes value in {'a', 'b'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly."
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(['a', 'b'], dtype='<U1'), [0.0, 1.0])
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array(['a', 'b'], dtype=object), [0.0, 1.0])
    msg = "y_true takes value in {b'a', b'b'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly."
    with pytest.raises(ValueError, match=msg):
        curve_func(np.array([b'a', b'b'], dtype='<S1'), [0.0, 1.0])
    y_pred = [0.0, 1.0, 0.2, 0.42]
    int_curve = curve_func([0, 1, 1, 0], y_pred)
    float_curve = curve_func([0.0, 1.0, 1.0, 0.0], y_pred)
    for int_curve_part, float_curve_part in zip(int_curve, float_curve):
        np.testing.assert_allclose(int_curve_part, float_curve_part)