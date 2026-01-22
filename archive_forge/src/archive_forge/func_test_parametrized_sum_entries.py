from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_parametrized_sum_entries(self, param_backend):
    """
        starting with a parametrized expression
        x1  x2
        [[[1  0],
         [0  0]],

         [[0  0],
         [0  1]]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along the row axis.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[[1  0]],

         [[0  1]]]
        """
    param_lin_op = linOpHelper((2,), type='param', data=2)
    param_backend.param_to_col = {2: 0, -1: 3}
    variable_lin_op = linOpHelper((2,), type='variable', data=1)
    var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
    mul_elem_lin_op = linOpHelper(data=param_lin_op)
    param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)
    sum_entries_lin_op = linOpHelper()
    out_view = param_backend.sum_entries(sum_entries_lin_op, param_var_view)
    out_repr = out_view.get_tensor_representation(0, 1)
    slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
    expected_idx_zero = np.array([[1.0, 0.0]])
    assert np.all(slice_idx_zero == expected_idx_zero)
    slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
    expected_idx_one = np.array([[0.0, 1.0]])
    assert np.all(slice_idx_one == expected_idx_one)
    assert out_view.get_tensor_representation(0, 1) == param_var_view.get_tensor_representation(0, 1)