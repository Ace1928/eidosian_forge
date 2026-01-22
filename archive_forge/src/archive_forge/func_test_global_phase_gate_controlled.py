import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('coeff, exp', [(-1, 1), (1j, 0.5), (-1j, -0.5), (1 / np.sqrt(2) * (1 + 1j), 0.25)])
def test_global_phase_gate_controlled(coeff, exp):
    g = cirq.GlobalPhaseGate(coeff)
    op = cirq.global_phase_operation(coeff)
    q = cirq.LineQubit.range(3)
    for num_controls, target_gate in zip(range(1, 4), [cirq.Z, cirq.CZ, cirq.CCZ]):
        assert g.controlled(num_controls) == target_gate ** exp
        np.testing.assert_allclose(cirq.unitary(cirq.ControlledGate(g, num_controls)), cirq.unitary(g.controlled(num_controls)))
        assert op.controlled_by(*q[:num_controls]) == target_gate(*q[:num_controls]) ** exp
    assert g.controlled(control_values=[0]) == cirq.ControlledGate(g, control_values=[0])
    xor_control_values = cirq.SumOfProducts(((0, 0), (1, 1)))
    assert g.controlled(control_values=xor_control_values) == cirq.ControlledGate(g, control_values=xor_control_values)