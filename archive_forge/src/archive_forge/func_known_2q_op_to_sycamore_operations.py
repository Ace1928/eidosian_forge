from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def known_2q_op_to_sycamore_operations(op: cirq.Operation) -> Optional[cirq.OP_TREE]:
    """Synthesizes a known two-qubit operation using `cirq_google.SYC` + single qubit rotations.

    This function dispatches to various known gate decompositions based on gate type. Currently,
    the following gates are known:
        1. Adjacent `cirq.SWAP` and `cirq.ZPowGate` wrapped in a circuit operation of length 2.
        2. `cirq.PhasedISwapPowGate` with exponent = 1 or phase_exponent = 0.25.
        3. `cirq.SWAP`, `cirq.ISWAP`.
        4. `cirq.CNotPowGate`, `cirq.CZPowGate`, `cirq.ZZPowGate`.

    Args:
        op: Operation to decompose.

    Returns:
        - A `cirq.OP_TREE` that implements the given known operation using only `cirq_google.SYC` +
        single qubit rotations OR
        - None if `op` is not a known operation.
    """
    if not (cirq.has_unitary(op) and cirq.num_qubits(op) == 2):
        return None
    q0, q1 = op.qubits
    if isinstance(op.untagged, cirq.CircuitOperation):
        flattened_gates = [o.gate for o in cirq.decompose_once(op.untagged)]
        if len(flattened_gates) != 2:
            return None
        for g1, g2 in itertools.permutations(flattened_gates):
            if g1 == cirq.SWAP and isinstance(g2, cirq.ZZPowGate):
                return _swap_rzz(g2.exponent * np.pi / 2, q0, q1)
    gate = op.gate
    if isinstance(gate, cirq.PhasedISwapPowGate):
        if math.isclose(gate.exponent, 1) and isinstance(gate.phase_exponent, float):
            return _decompose_phased_iswap_into_syc(gate.phase_exponent, q0, q1)
        if math.isclose(gate.phase_exponent, 0.25):
            return _decompose_phased_iswap_into_syc_precomputed(gate.exponent * np.pi / 2, q0, q1)
        return None
    if isinstance(gate, cirq.CNotPowGate):
        return [cirq.Y(q1) ** (-0.5), _decompose_cphase_into_syc(gate.exponent * np.pi, q0, q1), cirq.Y(q1) ** 0.5]
    if isinstance(gate, cirq.CZPowGate):
        return _decompose_cz_into_syc(q0, q1) if math.isclose(gate.exponent, 1) else _decompose_cphase_into_syc(gate.exponent * np.pi, q0, q1)
    if isinstance(gate, cirq.SwapPowGate) and math.isclose(gate.exponent, 1):
        return _decompose_swap_into_syc(q0, q1)
    if isinstance(gate, cirq.ISwapPowGate) and math.isclose(gate.exponent, 1):
        return _decompose_iswap_into_syc(q0, q1)
    if isinstance(gate, cirq.ZZPowGate):
        return _rzz(gate.exponent * np.pi / 2, q0, q1)
    return None