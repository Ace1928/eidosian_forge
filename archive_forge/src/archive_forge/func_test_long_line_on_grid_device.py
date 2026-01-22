import networkx as nx
import pytest
import cirq
def test_long_line_on_grid_device():
    step_circuit = construct_step_circuit(49)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(step_circuit)
    assert set(mapping.keys()) == set(step_circuit.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))
    device.validate_circuit(step_circuit.transform_qubits(mapping))
    step_circuit = construct_step_circuit(50)
    with pytest.raises(ValueError, match='No available physical qubits left on the device'):
        mapper.initial_mapping(step_circuit)