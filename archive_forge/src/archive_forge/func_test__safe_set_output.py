import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test__safe_set_output():
    """Check _safe_set_output works as expected."""
    est = EstimatorWithoutSetOutputAndWithoutTransform()
    _safe_set_output(est, transform='pandas')
    est = EstimatorNoSetOutputWithTransform()
    with pytest.raises(ValueError, match='Unable to configure output'):
        _safe_set_output(est, transform='pandas')
    est = EstimatorWithSetOutput().fit(np.asarray([[1, 2, 3]]))
    _safe_set_output(est, transform='pandas')
    config = _get_output_config('transform', est)
    assert config['dense'] == 'pandas'
    _safe_set_output(est, transform='default')
    config = _get_output_config('transform', est)
    assert config['dense'] == 'default'
    _safe_set_output(est, transform=None)
    config = _get_output_config('transform', est)
    assert config['dense'] == 'default'