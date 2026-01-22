import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def matrix_solve_with_broadcast(matrix, rhs, adjoint=False, name=None):
    """Solve systems of linear equations."""
    with ops.name_scope(name, 'MatrixSolveWithBroadcast', [matrix, rhs]):
        matrix = tensor_conversion.convert_to_tensor_v2_with_dispatch(matrix, name='matrix')
        rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(rhs, name='rhs', dtype=matrix.dtype)
        matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(matrix, rhs, adjoint_a=adjoint)
        matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])
        solution = linalg_ops.matrix_solve(matrix, rhs, adjoint=adjoint and still_need_to_transpose)
        return reshape_inv(solution)