import networkx as nx
import pytest
import cirq
def test_small_circuit_on_ring_device():
    circuit = construct_small_circuit()
    device_graph = cirq.testing.construct_ring_device(10, directed=True).metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(circuit)
    assert mapper.center == cirq.LineQubit(0)
    expected_circuit = cirq.Circuit([cirq.Moment(cirq.CNOT(cirq.LineQubit(2), cirq.LineQubit(1))), cirq.Moment(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))), cirq.Moment(cirq.CNOT(cirq.LineQubit(3), cirq.LineQubit(1)), cirq.X(cirq.LineQubit(4)))])
    cirq.testing.assert_same_circuits(circuit.transform_qubits(mapping), expected_circuit)