from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_parametrized_mul(self, param_backend):
    """
        Continuing from the non-parametrized example when the lhs is a parameter,
        instead of multiplying with known values, the matrix is split up into four slices,
        each representing an element of the parameter, i.e. instead of
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]

         we obtain the list of length four, where we have ones at the entries where previously
         we had the 1, 3, 2, and 4 (again flattened in column-major order):

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [0   0   0   0],
             [0   0   1   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [1   0   0   0],
             [0   0   0   0],
             [0   0   1   0]],

            [[0   1   0   0],
             [0   0   0   0],
             [0   0   0   1],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   1   0   0],
             [0   0   0   0],
             [0   0   0   1]]
        ]
        """
    variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
    param_backend.param_to_size = {-1: 1, 2: 4}
    param_backend.param_to_col = {2: 0, -1: 4}
    param_backend.param_size_plus_one = 5
    param_backend.var_length = 4
    view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 4)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
    assert np.all(view_A == np.eye(4))
    lhs_parameter = linOpHelper((2, 2), type='param', data=2)
    mul_lin_op = linOpHelper(data=lhs_parameter, args=[variable_lin_op])
    out_view = param_backend.mul(mul_lin_op, view)
    out_repr = out_view.get_tensor_representation(0, 4)
    slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
    expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert np.all(slice_idx_zero == expected_idx_zero)
    slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
    expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    assert np.all(slice_idx_one == expected_idx_one)
    slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
    expected_idx_two = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert np.all(slice_idx_two == expected_idx_two)
    slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
    expected_idx_three = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert np.all(slice_idx_three == expected_idx_three)
    assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)