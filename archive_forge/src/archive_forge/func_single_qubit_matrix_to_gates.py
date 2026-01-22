import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def single_qubit_matrix_to_gates(mat: np.ndarray, tolerance: float=0) -> List[ops.Gate]:
    """Implements a single-qubit operation with few gates.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of gates that, when applied in order, perform the desired
            operation.
    """
    rotations = single_qubit_matrix_to_pauli_rotations(mat, tolerance)
    return [pauli ** ht for pauli, ht in rotations]