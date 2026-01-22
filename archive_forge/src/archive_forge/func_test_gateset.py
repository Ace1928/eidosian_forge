import pytest
import cirq
@pytest.mark.parametrize('op,expected', [(cirq.H(Q), False), (cirq.HPowGate(exponent=0.5)(Q), False), (cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.125)(Q), True), (cirq.XPowGate(exponent=0.5)(Q), True), (cirq.YPowGate(exponent=0.25)(Q), True), (cirq.ZPowGate(exponent=0.125)(Q), True), (cirq.CZPowGate(exponent=0.5)(Q, Q2), False), (cirq.CZ(Q, Q2), True), (cirq.CNOT(Q, Q2), True), (cirq.SWAP(Q, Q2), False), (cirq.ISWAP(Q, Q2), False), (cirq.CCNOT(Q, Q2, Q3), True), (cirq.CCZ(Q, Q2, Q3), True), (cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3), True), (cirq.ParallelGate(cirq.Y, num_copies=3)(Q, Q2, Q3), True), (cirq.ParallelGate(cirq.Z, num_copies=3)(Q, Q2, Q3), True), (cirq.X(Q).controlled_by(Q2, Q3), True), (cirq.Z(Q).controlled_by(Q2, Q3), True), (cirq.ZPowGate(exponent=0.5)(Q).controlled_by(Q2, Q3), False)])
def test_gateset(op: cirq.Operation, expected: bool):
    assert cirq.is_native_neutral_atom_op(op) == expected
    if op.gate is not None:
        assert cirq.is_native_neutral_atom_gate(op.gate) == expected