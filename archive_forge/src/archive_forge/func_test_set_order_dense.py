import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('input_order', ['C', 'F'])
def test_set_order_dense(order, input_order):
    """Check that _set_order returns arrays with promised order."""
    X = np.array([[0], [0], [0]], order=input_order)
    y = np.array([0, 0, 0], order=input_order)
    X2, y2 = _set_order(X, y, order=order)
    if order == 'C':
        assert X2.flags['C_CONTIGUOUS']
        assert y2.flags['C_CONTIGUOUS']
    elif order == 'F':
        assert X2.flags['F_CONTIGUOUS']
        assert y2.flags['F_CONTIGUOUS']
    if order == input_order:
        assert X is X2
        assert y is y2