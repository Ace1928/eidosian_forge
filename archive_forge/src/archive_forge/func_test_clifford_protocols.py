import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('gate_cls', (cirq.XXPowGate, cirq.YYPowGate, cirq.ZZPowGate))
@pytest.mark.parametrize('exponent,is_clifford', ((0, True), (0.5, True), (0.75, False), (1, True), (1.5, True), (-1.5, True)))
def test_clifford_protocols(gate_cls: type[cirq.EigenGate], exponent: float, is_clifford: bool):
    gate = gate_cls(exponent=exponent)
    assert hasattr(gate, '_decompose_into_clifford_with_qubits_')
    if is_clifford:
        clifford_decomposition = cirq.Circuit(gate._decompose_into_clifford_with_qubits_(cirq.LineQubit.range(2)))
        assert cirq.has_stabilizer_effect(gate)
        assert cirq.has_stabilizer_effect(clifford_decomposition)
        if exponent == 0:
            assert clifford_decomposition == cirq.Circuit()
        else:
            np.testing.assert_allclose(cirq.unitary(gate), cirq.unitary(clifford_decomposition))
    else:
        assert not cirq.has_stabilizer_effect(gate)
        assert gate._decompose_into_clifford_with_qubits_(cirq.LineQubit.range(2)) is NotImplemented