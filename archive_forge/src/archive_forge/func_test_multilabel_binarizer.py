import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer():
    inputs = [lambda: [(2, 3), (1,), (1, 2)], lambda: ({2, 3}, {1}, {1, 2}), lambda: iter([iter((2, 3)), iter((1,)), {1, 2}])]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    inverse = inputs[0]()
    for inp in inputs:
        mlb = MultiLabelBinarizer()
        got = mlb.fit_transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse
        mlb = MultiLabelBinarizer()
        got = mlb.fit(inp()).transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse