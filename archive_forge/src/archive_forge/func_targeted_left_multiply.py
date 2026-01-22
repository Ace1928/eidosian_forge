import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def targeted_left_multiply(left_matrix: np.ndarray, right_target: np.ndarray, target_axes: Sequence[int], out: Optional[np.ndarray]=None) -> np.ndarray:
    """Left-multiplies the given axes of the target tensor by the given matrix.

    Note that the matrix must have a compatible tensor structure.

    For example, if you have an 6-qubit state vector `input_state` with shape
    (2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation `op` with shape
    (2, 2, 2, 2), and you want to apply `op` to the 5'th and 3'rd qubits
    within `input_state`, then the output state vector is computed as follows:

        output_state = cirq.targeted_left_multiply(op, input_state, [5, 3])

    This method also works when the right hand side is a matrix instead of a
    vector. If a unitary circuit's matrix is `old_effect`, and you append
    a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
    at offset 1 and the target q4 is the qubit at offset 4, then the appended
    circuit's unitary matrix is computed as follows:

        new_effect = cirq.targeted_left_multiply(
            left_matrix=cirq.unitary(cirq.CNOT).reshape((2, 2, 2, 2)),
            right_target=old_effect,
            target_axes=[1, 4])

    Args:
        left_matrix: What to left-multiply the target tensor by.
        right_target: A tensor to carefully broadcast a left-multiply over.
        target_axes: Which axes of the target are being operated on.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as right_target.

    Returns:
        The output tensor.

    Raises:
        ValueError: If `out` is either `right_target` or `left_matrix`.
    """
    if out is right_target or out is left_matrix:
        raise ValueError('out is right_target or out is left_matrix')
    k = len(target_axes)
    d = len(right_target.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple((data_indices[q] for q in target_axes))
    input_indices = work_indices + used_data_indices
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_axes):
        output_indices[t] = w
    all_indices = set(input_indices + data_indices + tuple(output_indices))
    return np.einsum(left_matrix, input_indices, right_target, data_indices, output_indices, optimize=len(all_indices) >= 26, **{'out': out} if out is not None else {})