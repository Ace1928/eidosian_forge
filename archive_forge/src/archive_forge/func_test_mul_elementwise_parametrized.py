from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_mul_elementwise_parametrized(self, param_backend):
    """
        Continuing the non-parametrized example when 'a' is a parameter, instead of multiplying
        with known values, the matrix is split up into two slices, each representing an element
        of the parameter, i.e. instead of
         x1  x2
        [[2  0],
         [0  3]]

         we obtain the list of length two:

          x1  x2
        [
         [[1  0],
          [0  0]],

         [[0  0],
          [0  1]]
        ]
        """
    variable_lin_op = linOpHelper((2,), type='variable', data=1)
    view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 2)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
    assert np.all(view_A == np.eye(2))
    lhs_parameter = linOpHelper((2,), type='param', data=2)
    mul_elementwise_lin_op = linOpHelper(data=lhs_parameter)
    out_view = param_backend.mul_elem(mul_elementwise_lin_op, view)
    out_repr = out_view.get_tensor_representation(0, 2)
    slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
    expected_idx_zero = np.array([[1, 0], [0, 0]])
    assert np.all(slice_idx_zero == expected_idx_zero)
    slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
    expected_idx_one = np.array([[0, 0], [0, 1]])
    assert np.all(slice_idx_one == expected_idx_one)
    assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)