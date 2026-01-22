import pytest
import cirq
def test_hardcoded_initial_mapper():
    input_map = {cirq.NamedQubit(str(i)): cirq.NamedQubit(str(-i)) for i in range(1, 6)}
    circuit = cirq.Circuit([cirq.H(cirq.NamedQubit(str(i))) for i in range(1, 6)])
    initial_mapper = cirq.HardCodedInitialMapper(input_map)
    assert input_map == initial_mapper.initial_mapping(circuit)
    assert str(initial_mapper) == f'cirq.HardCodedInitialMapper({input_map})'
    cirq.testing.assert_equivalent_repr(initial_mapper)
    circuit.append(cirq.H(cirq.NamedQubit(str(6))))
    with pytest.raises(ValueError, match='The qubits in circuit must be a subset of the keys in the mapping'):
        initial_mapper.initial_mapping(circuit)