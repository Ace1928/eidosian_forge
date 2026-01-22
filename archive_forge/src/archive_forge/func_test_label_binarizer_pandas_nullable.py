import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
@pytest.mark.parametrize('unique_first', [True, False])
def test_label_binarizer_pandas_nullable(dtype, unique_first):
    """Checks that LabelBinarizer works with pandas nullable dtypes.

    Non-regression test for gh-25637.
    """
    pd = pytest.importorskip('pandas')
    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    if unique_first:
        y_true = y_true.unique()
    lb = LabelBinarizer().fit(y_true)
    y_out = lb.transform([1, 0])
    assert_array_equal(y_out, [[1], [0]])