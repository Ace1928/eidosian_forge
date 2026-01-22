import pytest
import cirq
def test_op_calls_validate():
    q0, q1, q2 = _make_qubits(3)
    bad_qubit = cirq.NamedQubit('bad')

    class ValidError(Exception):
        pass

    class ValiGate(cirq.PauliStringGateOperation):

        def validate_args(self, qubits):
            super().validate_args(qubits)
            if bad_qubit in qubits:
                raise ValidError()

        def map_qubits(self, qubit_map):
            ps = self.pauli_string.map_qubits(qubit_map)
            return ValiGate(ps)
    g = ValiGate(cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}))
    _ = g.with_qubits(q1, q0, q2)
    with pytest.raises(ValidError):
        _ = g.with_qubits(q0, q1, bad_qubit)