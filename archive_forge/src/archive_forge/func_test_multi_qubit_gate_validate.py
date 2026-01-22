import pytest
import cirq
def test_multi_qubit_gate_validate():

    class Example(cirq.Gate):

        def _num_qubits_(self) -> int:
            return self._num_qubits

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits
    a, b, c, d = cirq.LineQubit.range(4)
    g = Example(3)
    assert g.num_qubits() == 3
    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])