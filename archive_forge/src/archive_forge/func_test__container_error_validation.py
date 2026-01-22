import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test__container_error_validation(csr_container):
    """Check errors in _wrap_data_with_container."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    X_csr = csr_container(X)
    match = 'The transformer outputs a scipy sparse matrix.'
    with config_context(transform_output='pandas'):
        with pytest.raises(ValueError, match=match):
            _wrap_data_with_container('transform', X_csr, X, StandardScaler())