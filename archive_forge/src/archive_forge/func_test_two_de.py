from unittest.mock import Mock
import numpy as np
import pytest
import scipy.stats as st
from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
from ..helpers import RandomVariableTestClass
@pytest.mark.parametrize('data', [np.random.randn(1000), np.random.randn(1000).tolist()])
def test_two_de(data):
    """Test to check for custom atleast_2d. List added to test for a non ndarray case."""
    assert np.allclose(two_de(data), np.atleast_2d(data))