import pytest
import cirq
import cirq_pasqal
@pytest.mark.parametrize('op,expected', [(cirq.H(Q), True), (cirq.HPowGate(exponent=0.5)(Q), False), (cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.125)(Q), True), (cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3), True), (cirq.CZPowGate(exponent=0.5)(Q, Q2), False), (cirq.CZ(Q, Q2), True), (cirq.CNOT(Q, Q2), False), (cirq.CCNOT(Q, Q2, Q3), False), (cirq.CCZ(Q, Q2, Q3), False), (cirq.Z(Q).controlled_by(Q2), True), (cirq.X(Q).controlled_by(Q2, Q3), False), (cirq.Z(Q).controlled_by(Q2, Q3), False), (cirq.ZPowGate(exponent=0.5)(Q).controlled_by(Q2, Q3), False)])
def test_control_gates_not_included(op: cirq.Operation, expected: bool):
    gs = cirq_pasqal.PasqalGateset(include_additional_controlled_ops=False)
    assert gs.validate(op) == expected
    assert gs.validate(cirq.Circuit(op)) == expected