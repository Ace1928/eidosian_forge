import pytest
import cirq
def test_two_qubit_gate_validate_pass():

    class Example(cirq.testing.TwoQubitGate):

        def matrix(self):
            pass
    g = Example()
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    assert g.num_qubits() == 2
    g.validate_args([q1, q2])
    g.validate_args([q2, q3])
    g.validate_args([q3, q2])