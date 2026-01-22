import pytest
import sympy
import cirq
from cirq.ops.gateset_test import CustomX, CustomXPowGate
def test_any_unitary_gate_family():
    with pytest.raises(ValueError, match='must be a positive integer'):
        _ = cirq.AnyUnitaryGateFamily(0)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.AnyUnitaryGateFamily())
    for num_qubits in range(1, 6, 2):
        q = cirq.LineQubit.range(num_qubits)
        gate = UnitaryGate(num_qubits)
        eq.add_equality_group(cirq.AnyUnitaryGateFamily(num_qubits))
        for init_num_qubits in [None, num_qubits]:
            gate_family = cirq.AnyUnitaryGateFamily(init_num_qubits)
            cirq.testing.assert_equivalent_repr(gate_family)
            assert gate in gate_family
            assert gate(*q) in gate_family
            if init_num_qubits:
                assert f'{init_num_qubits}' in gate_family.name
                assert f'{init_num_qubits}' in gate_family.description
                assert UnitaryGate(num_qubits + 1) not in gate_family
            else:
                assert 'Any-Qubit' in gate_family.name
                assert 'any unitary' in gate_family.description
    assert cirq.testing.SingleQubitGate() not in cirq.AnyUnitaryGateFamily()