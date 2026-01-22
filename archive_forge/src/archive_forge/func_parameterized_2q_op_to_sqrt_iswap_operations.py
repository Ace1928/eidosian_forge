from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def parameterized_2q_op_to_sqrt_iswap_operations(op: 'cirq.Operation', *, use_sqrt_iswap_inv: bool=False) -> protocols.decompose_protocol.DecomposeResult:
    """Tries to decompose a parameterized 2q operation into √iSWAP's + parameterized 1q rotations.

    Currently only supports decomposing the following gates:
        a) `cirq.CZPowGate`
        b) `cirq.SwapPowGate`
        c) `cirq.ISwapPowGate`
        d) `cirq.FSimGate`

    Args:
        op: Parameterized two qubit operation to be decomposed into sqrt-iswaps.
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used as the target 2q gate, instead
            of `cirq.SQRT_ISWAP`.

    Returns:
        A parameterized `cirq.OP_TREE` implementing `op` using only `cirq.SQRT_ISWAP`
        (or `cirq.SQRT_ISWAP_INV`) and parameterized single qubit rotations OR
        None or NotImplemented if decomposition of `op` is not known.
    """
    gate = op.gate
    q0, q1 = op.qubits
    if isinstance(gate, ops.CZPowGate):
        return _cphase_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.SwapPowGate):
        return _swap_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.ISwapPowGate):
        return _iswap_symbols_to_sqrt_iswap(q0, q1, gate.exponent, use_sqrt_iswap_inv)
    if isinstance(gate, ops.FSimGate):
        return _fsim_symbols_to_sqrt_iswap(q0, q1, gate.theta, gate.phi, use_sqrt_iswap_inv)
    return NotImplemented