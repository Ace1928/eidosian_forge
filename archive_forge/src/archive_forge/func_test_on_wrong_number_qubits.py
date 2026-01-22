import pytest
import cirq
def test_on_wrong_number_qubits():
    q0, q1, q2 = _make_qubits(3)

    class ExampleGate(cirq.PauliStringGateOperation):

        def map_qubits(self, qubit_map):
            ps = self.pauli_string.map_qubits(qubit_map)
            return ExampleGate(ps)
    g = ExampleGate(cirq.PauliString({q0: cirq.X, q1: cirq.Y}))
    _ = g.with_qubits(q1, q2)
    with pytest.raises(ValueError):
        _ = g.with_qubits()
    with pytest.raises(ValueError):
        _ = g.with_qubits(q2)
    with pytest.raises(ValueError):
        _ = g.with_qubits(q0, q1, q2)