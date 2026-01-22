from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def two_qubit_matrix_to_cz_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, allow_partial_czs: bool, atol: float=1e-08, clean_operations: bool=True) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into Z/XY/CZ gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        clean_operations: Enables optimizing resulting operation list by
            merging operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the matrix.
    """
    kak = linalg.kak_decomposition(mat, atol=atol)
    operations = _kak_decomposition_to_operations(q0, q1, kak, allow_partial_czs, atol=atol)
    if clean_operations:
        return cleanup_operations(operations)
    return operations