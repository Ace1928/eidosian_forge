import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_set_output_pandas_keep_index():
    """Check that set_output does not override index.

    Non-regression test for gh-25730.
    """
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])
    est = EstimatorWithSetOutputIndex().set_output(transform='pandas')
    est.fit(X)
    X_trans = est.transform(X)
    assert_array_equal(X_trans.index, ['s0', 's1'])