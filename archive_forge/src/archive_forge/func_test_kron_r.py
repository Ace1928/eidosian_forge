from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
def test_kron_r(self, backend):
    """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1],
             [2]],

        kron(a, x) means we have
        [[x11, x12],
         [x21, x22],
         [2x11, 2x12],
         [2x21, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   2   0],
         [0   0   0   2]]

        However computing kron(a, x) (where x is represented as eye(4))
        directly gives us:
        [[1   0   0   0],
         [2   0   0   0],
         [0   1   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   2   0],
         [0   0   0   1],
         [0   0   0   2]]
        So we must swap the row indices of the resulting matrix.
        """
    variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
    view = backend.process_constraint(variable_lin_op, backend.get_empty_view())
    view_A = view.get_tensor_representation(0, 4)
    view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
    assert np.all(view_A == np.eye(4))
    a = linOpHelper((2, 1), type='dense_const', data=np.array([[1], [2]]))
    kron_r_lin_op = linOpHelper(data=a, args=[variable_lin_op])
    out_view = backend.kron_r(kron_r_lin_op, view)
    A = out_view.get_tensor_representation(0, 8)
    A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
    expected = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    assert np.all(A == expected)
    assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)