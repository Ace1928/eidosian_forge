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
def loss_instance_name(param):
    if isinstance(param, BaseLoss):
        loss = param
        name = loss.__class__.__name__
        if isinstance(loss, PinballLoss):
            name += f'(quantile={loss.closs.quantile})'
        elif isinstance(loss, HuberLoss):
            name += f'(quantile={loss.quantile}'
        elif hasattr(loss, 'closs') and hasattr(loss.closs, 'power'):
            name += f'(power={loss.closs.power})'
        return name
    else:
        return str(param)