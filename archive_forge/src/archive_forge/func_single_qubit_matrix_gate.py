from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def single_qubit_matrix_gate(matrix: Optional[np.ndarray]) -> Optional[QuirkOp]:
    if matrix is None or matrix.shape[0] != 2:
        return None
    matrix = matrix.round(6)
    matrix_repr = '{{%s+%si,%s+%si},{%s+%si,%s+%si}}' % (np.real(matrix[0, 0]), np.imag(matrix[0, 0]), np.real(matrix[1, 0]), np.imag(matrix[1, 0]), np.real(matrix[0, 1]), np.imag(matrix[0, 1]), np.real(matrix[1, 1]), np.imag(matrix[1, 1]))
    matrix_repr = matrix_repr.replace('+-', '-')
    matrix_repr = matrix_repr.replace('+0.0i', '')
    matrix_repr = matrix_repr.replace('.0,', ',')
    matrix_repr = matrix_repr.replace('.0}', '}')
    matrix_repr = matrix_repr.replace('.0+', '+')
    matrix_repr = matrix_repr.replace('.0-', '-')
    return QuirkOp({'id': '?', 'matrix': matrix_repr})