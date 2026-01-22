import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_auto_wrap_output_keys_errors_with_incorrect_input():
    msg = 'auto_wrap_output_keys must be None or a tuple of keys.'
    with pytest.raises(ValueError, match=msg):

        class BadEstimator(_SetOutputMixin, auto_wrap_output_keys='bad_parameter'):
            pass