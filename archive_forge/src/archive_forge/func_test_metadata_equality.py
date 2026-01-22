import networkx as nx
import cirq
def test_metadata_equality():
    qubits = cirq.LineQubit.range(4)
    graph = nx.star_graph(3)
    graph2 = nx.star_graph(3)
    graph.add_edge(1, 2, directed=False)
    graph2.add_edge(1, 2, directed=True)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.DeviceMetadata(qubits, graph))
    eq.add_equality_group(cirq.DeviceMetadata(qubits, graph2))
    eq.add_equality_group(cirq.DeviceMetadata(qubits[1:], graph))