import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('arr_type', [np.array] + COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_label_binarize_multilabel(arr_type):
    y_ind = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = pos_label * y_ind
    y = arr_type(y_ind)
    check_binarized_results(y, classes, pos_label, neg_label, expected)
    with pytest.raises(ValueError):
        label_binarize(y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True)