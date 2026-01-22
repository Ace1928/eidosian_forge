import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test__safe_set_output_error():
    """Check transform with invalid config."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput()
    _safe_set_output(est, transform='bad')
    msg = 'output config must be in'
    with pytest.raises(ValueError, match=msg):
        est.transform(X)