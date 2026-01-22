import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('encoder', [LabelEncoder(), LabelBinarizer(), MultiLabelBinarizer()])
def test_label_encoders_do_not_have_set_output(encoder):
    """Check that label encoders do not define set_output and work with y as a kwarg.

    Non-regression test for #26854.
    """
    assert not hasattr(encoder, 'set_output')
    y_encoded_with_kwarg = encoder.fit_transform(y=['a', 'b', 'c'])
    y_encoded_positional = encoder.fit_transform(['a', 'b', 'c'])
    assert_array_equal(y_encoded_with_kwarg, y_encoded_positional)