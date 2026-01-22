import cirq
from cirq.ops.named_qubit import _pad_digits
def test_named_qubit_range():
    qubits = cirq.NamedQubit.range(2, prefix='a')
    assert qubits == [cirq.NamedQubit('a0'), cirq.NamedQubit('a1')]
    qubits = cirq.NamedQubit.range(-1, 4, 2, prefix='a')
    assert qubits == [cirq.NamedQubit('a-1'), cirq.NamedQubit('a1'), cirq.NamedQubit('a3')]