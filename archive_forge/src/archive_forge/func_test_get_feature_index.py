import numpy as np
import pytest
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('fx, idx', [(0, 0), (1, 1), ('a', 0), ('b', 1), ('c', 2)])
def test_get_feature_index(fx, idx):
    feature_names = ['a', 'b', 'c']
    assert _get_feature_index(fx, feature_names) == idx