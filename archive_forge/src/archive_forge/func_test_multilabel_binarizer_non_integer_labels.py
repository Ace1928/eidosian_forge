import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer_non_integer_labels():
    tuple_classes = _to_object_array([(1,), (2,), (3,)])
    inputs = [([('2', '3'), ('1',), ('1', '2')], ['1', '2', '3']), ([('b', 'c'), ('a',), ('a', 'b')], ['a', 'b', 'c']), ([((2,), (3,)), ((1,),), ((1,), (2,))], tuple_classes)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    for inp, classes in inputs:
        mlb = MultiLabelBinarizer()
        inp = np.array(inp, dtype=object)
        assert_array_equal(mlb.fit_transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)
        mlb = MultiLabelBinarizer()
        assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)
    mlb = MultiLabelBinarizer()
    with pytest.raises(TypeError):
        mlb.fit_transform([{}, ({}, {'a': 'b'})])