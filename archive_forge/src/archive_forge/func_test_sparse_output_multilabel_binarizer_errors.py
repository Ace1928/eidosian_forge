import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_output_multilabel_binarizer_errors(csr_container):
    inp = iter([iter((2, 3)), iter((1,)), {1, 2}])
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(inp)
    with pytest.raises(ValueError):
        mlb.inverse_transform(csr_container(np.array([[0, 1, 1], [2, 0, 0], [1, 1, 0]])))