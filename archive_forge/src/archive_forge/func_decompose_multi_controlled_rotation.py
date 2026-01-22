from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def decompose_multi_controlled_rotation(matrix: np.ndarray, controls: List['cirq.Qid'], target: 'cirq.Qid') -> List['cirq.Operation']:
    """Implements action of multi-controlled unitary gate.

    Returns a sequence of operations, which is equivalent to applying
    single-qubit gate with matrix `matrix` on `target`, controlled by
    `controls`.

    Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
    gates.

    If matrix is special unitary, result has length `O(len(controls))`.
    Otherwise result has length `O(len(controls)**2)`.

    References:
        [1] Barenco, Bennett et al.
            Elementary gates for quantum computation. 1995.
            https://arxiv.org/pdf/quant-ph/9503016.pdf

    Args:
        matrix - 2x2 numpy unitary matrix (of real or complex dtype).
        controls - control qubits.
        targets - target qubits.

    Returns:
        A list of operations which, applied in a sequence, are equivalent to
        applying `MatrixGate(matrix).on(target).controlled_by(*controls)`.
    """
    assert is_unitary(matrix)
    assert matrix.shape == (2, 2)
    if len(controls) == 0:
        return [ops.MatrixGate(matrix).on(target)]
    elif len(controls) == 1:
        return _decompose_single_ctrl(matrix, controls[0], target)
    elif is_special_unitary(matrix):
        return _decompose_su(matrix, controls, target)
    else:
        return _decompose_recursive(matrix, 1.0, controls, target, [])