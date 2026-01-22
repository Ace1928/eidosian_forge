import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
from scipy.special import logsumexp
from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
from sklearn.utils import _IS_WASM, assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit
@pytest.mark.parametrize('loss, params, err_type, err_msg', [(PinballLoss, {'quantile': None}, TypeError, 'quantile must be an instance of float, not NoneType.'), (PinballLoss, {'quantile': 0}, ValueError, 'quantile == 0, must be > 0.'), (PinballLoss, {'quantile': 1.1}, ValueError, 'quantile == 1.1, must be < 1.'), (HuberLoss, {'quantile': None}, TypeError, 'quantile must be an instance of float, not NoneType.'), (HuberLoss, {'quantile': 0}, ValueError, 'quantile == 0, must be > 0.'), (HuberLoss, {'quantile': 1.1}, ValueError, 'quantile == 1.1, must be < 1.')])
def test_loss_init_parameter_validation(loss, params, err_type, err_msg):
    """Test that loss raises errors for invalid input."""
    with pytest.raises(err_type, match=err_msg):
        loss(**params)