from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_tensor_representation():
    A = TensorRepresentation(np.array([10]), np.array([0]), np.array([1]), np.array([0]), shape=(2, 2))
    B = TensorRepresentation(np.array([20]), np.array([1]), np.array([1]), np.array([1]), shape=(2, 2))
    combined = TensorRepresentation.combine([A, B])
    assert np.all(combined.data == np.array([10, 20]))
    assert np.all(combined.row == np.array([0, 1]))
    assert np.all(combined.col == np.array([1, 1]))
    assert np.all(combined.parameter_offset == np.array([0, 1]))
    assert combined.shape == (2, 2)
    flattened = combined.flatten_tensor(2)
    assert np.all(flattened.toarray() == np.array([[0, 0], [0, 0], [10, 0], [0, 20]]))