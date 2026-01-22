import pytest
import cirq
def test_two_qubit_gate_validate_wrong_number():

    class Example(cirq.testing.TwoQubitGate):

        def matrix(self):
            pass
    g = Example()
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2, q3])