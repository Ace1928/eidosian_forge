import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@ignore_warnings
def test_label_binarizer_errors():
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)
    multi_label = [(2, 3), (0,), (0, 2)]
    err_msg = 'You appear to be using a legacy multi-label data representation.'
    with pytest.raises(ValueError, match=err_msg):
        lb.transform(multi_label)
    lb = LabelBinarizer()
    err_msg = 'This LabelBinarizer instance is not fitted yet'
    with pytest.raises(ValueError, match=err_msg):
        lb.transform([])
    with pytest.raises(ValueError, match=err_msg):
        lb.inverse_transform([])
    input_labels = [0, 1, 0, 1]
    err_msg = 'neg_label=2 must be strictly less than pos_label=1.'
    lb = LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = 'neg_label=2 must be strictly less than pos_label=2.'
    lb = LabelBinarizer(neg_label=2, pos_label=2)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = 'Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label=2 and neg_label=1'
    lb = LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    y_seq_of_seqs = [[], [1, 2], [3], [0, 1, 3], [2]]
    err_msg = 'You appear to be using a legacy multi-label data representation'
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit_transform(y_seq_of_seqs)
    err_msg = "output_type='binary', but y.shape"
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(y=np.array([[1, 2, 3], [2, 1, 3]]), output_type='binary', classes=[1, 2, 3], threshold=0)
    err_msg = 'Multioutput target data is not supported with label binarization'
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    with pytest.raises(ValueError, match=err_msg):
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])