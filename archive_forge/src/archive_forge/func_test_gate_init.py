import numpy as np
import pytest
import sympy
import cirq
def test_gate_init():
    gate = cirq.GlobalPhaseGate(1j)
    assert gate.coefficient == 1j
    assert isinstance(gate.on(), cirq.GateOperation)
    assert gate.on().gate == gate
    assert cirq.has_stabilizer_effect(gate)
    with pytest.raises(ValueError, match='Coefficient is not unitary'):
        _ = cirq.GlobalPhaseGate(2)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        _ = gate.on(cirq.LineQubit(0))