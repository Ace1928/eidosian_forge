from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_get_variable_tensor(self, scipy_backend):
    outer = scipy_backend.get_variable_tensor((2,), 1)
    assert outer.keys() == {1}, 'Should only be in variable with ID 1'
    inner = outer[1]
    assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
    tensor = inner[-1]
    assert isinstance(tensor, sp.spmatrix), 'Should be a scipy sparse matrix'
    assert tensor.shape == (2, 2), 'Should be a 1*2x2 tensor'
    assert np.all(tensor == np.eye(2)), 'Should be eye(2)'