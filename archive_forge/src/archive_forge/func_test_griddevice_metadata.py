import pytest
import cirq
import networkx as nx
def test_griddevice_metadata():
    qubits = cirq.GridQubit.rect(2, 3)
    qubit_pairs = [(a, b) for a in qubits for b in qubits if a != b and a.is_adjacent(b)]
    isolated_qubits = [cirq.GridQubit(9, 9), cirq.GridQubit(10, 10)]
    gateset = cirq.Gateset(cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZ)
    gate_durations = {cirq.GateFamily(cirq.XPowGate): 1000, cirq.GateFamily(cirq.YPowGate): 1000, cirq.GateFamily(cirq.ZPowGate): 1000}
    target_gatesets = (cirq.CZTargetGateset(),)
    metadata = cirq.GridDeviceMetadata(qubit_pairs, gateset, gate_durations=gate_durations, all_qubits=qubits + isolated_qubits, compilation_target_gatesets=target_gatesets)
    expected_pairings = frozenset({frozenset((cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))), frozenset((cirq.GridQubit(0, 1), cirq.GridQubit(0, 2))), frozenset((cirq.GridQubit(0, 1), cirq.GridQubit(1, 1))), frozenset((cirq.GridQubit(0, 2), cirq.GridQubit(1, 2))), frozenset((cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))), frozenset((cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))), frozenset((cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)))})
    assert metadata.qubit_set == frozenset(qubits + isolated_qubits)
    assert metadata.qubit_pairs == expected_pairings
    assert metadata.gateset == gateset
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from(sorted(list(qubits + isolated_qubits)))
    expected_graph.add_edges_from(sorted(list(expected_pairings)), directed=False)
    assert metadata.nx_graph.edges() == expected_graph.edges()
    assert metadata.nx_graph.nodes() == expected_graph.nodes()
    assert metadata.gate_durations == gate_durations
    assert metadata.isolated_qubits == frozenset(isolated_qubits)
    assert metadata.compilation_target_gatesets == target_gatesets