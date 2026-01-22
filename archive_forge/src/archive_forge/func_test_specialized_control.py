import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
@pytest.mark.parametrize('input_gate, specialized_output', [(cirq.Z, cirq.CZ), (cirq.CZ, cirq.CCZ), (cirq.X, cirq.CX), (cirq.CX, cirq.CCX), (cirq.ZPowGate(exponent=0.5), cirq.CZPowGate(exponent=0.5)), (cirq.CZPowGate(exponent=0.5), cirq.CCZPowGate(exponent=0.5)), (cirq.XPowGate(exponent=0.5), cirq.CXPowGate(exponent=0.5)), (cirq.CXPowGate(exponent=0.5), cirq.CCXPowGate(exponent=0.5))])
def test_specialized_control(input_gate, specialized_output):
    assert input_gate.controlled() == specialized_output
    assert input_gate.controlled(num_controls=1) == specialized_output
    assert input_gate.controlled(control_values=((1,),)) == specialized_output
    assert input_gate.controlled(control_values=cirq.SumOfProducts([[1]])) == specialized_output
    assert input_gate.controlled(control_qid_shape=(2,)) == specialized_output
    assert np.allclose(cirq.unitary(specialized_output), cirq.unitary(cirq.ControlledGate(input_gate, num_controls=1)))
    assert input_gate.controlled().controlled() == specialized_output.controlled(num_controls=1)
    assert input_gate.controlled(num_controls=2) == specialized_output.controlled(num_controls=1)
    assert input_gate.controlled(control_values=((0,), (0,), (1,))) == specialized_output.controlled(num_controls=2, control_values=((0,), (0,)))
    assert input_gate.controlled(control_qid_shape=(3, 3, 2)) == specialized_output.controlled(num_controls=2, control_qid_shape=(3, 3))
    assert input_gate.controlled(control_qid_shape=(2,)).controlled(control_qid_shape=(3,)).controlled(control_qid_shape=(4,)) != specialized_output.controlled(num_controls=2, control_qid_shape=(3, 4))
    assert input_gate.controlled(num_controls=1, control_qid_shape=(3,)) == cirq.ControlledGate(input_gate, num_controls=1, control_qid_shape=(3,))
    assert input_gate.controlled(control_values=((0,), (1,), (0,))) == cirq.ControlledGate(input_gate, num_controls=3, control_values=((0,), (1,), (0,)))
    assert input_gate.controlled(control_qid_shape=(3, 2, 3)) == cirq.ControlledGate(input_gate, num_controls=3, control_qid_shape=(3, 2, 3))
    assert input_gate.controlled(control_qid_shape=(3,)).controlled(control_qid_shape=(2,)).controlled(control_qid_shape=(4,)) != cirq.ControlledGate(input_gate, num_controls=3, control_qid_shape=(3, 2, 4))