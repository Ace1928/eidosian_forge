from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
def test_var_names_warning():
    """Test confusing var_name handling"""
    data = from_dict(posterior={'~mu': np.random.randn(2, 10), 'mu': -np.random.randn(2, 10), 'theta': np.random.randn(2, 10, 8)}).posterior
    var_names = expected = ['~mu']
    with pytest.warns(UserWarning):
        assert _var_names(var_names, data) == expected