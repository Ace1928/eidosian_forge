from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def two_qubit_matrix_to_sqrt_iswap_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', mat: np.ndarray, *, required_sqrt_iswap_count: Optional[int]=None, use_sqrt_iswap_inv: bool=False, atol: float=1e-08, check_preconditions: bool=True, clean_operations: bool=False) -> Sequence['cirq.Operation']:
    """Decomposes a two-qubit operation into ZPow/XPow/YPow/sqrt-iSWAP gates.

    This method uses the KAK decomposition of the matrix to determine how many
    sqrt-iSWAP gates are needed and which single-qubit gates to use in between
    each sqrt-iSWAP.

    All operations can be synthesized with exactly three sqrt-iSWAP gates and
    about 79% of operations (randomly chosen under the Haar measure) can also be
    synthesized with two sqrt-iSWAP gates.  Only special cases locally
    equivalent to identity or sqrt-iSWAP can be synthesized with zero or one
    sqrt-iSWAP gates respectively.  Unless ``required_sqrt_iswap_count`` is
    specified, the fewest possible number of sqrt-iSWAP will be used.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        required_sqrt_iswap_count: When specified, exactly this many sqrt-iSWAP
            gates will be used even if fewer is possible (maximum 3).  Raises
            ``ValueError`` if impossible.
        use_sqrt_iswap_inv: If True, returns a decomposition using
            ``SQRT_ISWAP_INV`` gates instead of ``SQRT_ISWAP``.  This
            decomposition is identical except for the addition of single-qubit
            Z gates.
        atol: A limit on the amount of absolute error introduced by the
            construction.
        check_preconditions: If set, verifies that the input corresponds to a
            4x4 unitary before decomposing.
        clean_operations: Merges runs of single qubit gates to a single `cirq.PhasedXZGate` in
            the resulting operations list.

    Returns:
        A list of operations implementing the matrix including at most three
        ``SQRT_ISWAP`` (sqrt-iSWAP) gates and ZPow, XPow, and YPow single-qubit
        gates.

    Raises:
        ValueError:
            If ``required_sqrt_iswap_count`` is specified, the minimum number of
            sqrt-iSWAP gates needed to decompose the given matrix is greater
            than ``required_sqrt_iswap_count``.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    kak = linalg.kak_decomposition(mat, atol=atol / 10, rtol=0, check_preconditions=check_preconditions)
    operations = _kak_decomposition_to_sqrt_iswap_operations(q0, q1, kak, required_sqrt_iswap_count, use_sqrt_iswap_inv, atol=atol)
    return [*merge_single_qubit_gates_to_phxz(circuits.Circuit(operations)).all_operations()] if clean_operations else operations