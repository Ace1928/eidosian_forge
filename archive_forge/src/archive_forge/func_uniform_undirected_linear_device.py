from typing import Any, Dict, Hashable, Iterable, Mapping, Optional
from cirq import devices, ops
from cirq.contrib.graph_device.graph_device import UndirectedGraphDevice, UndirectedGraphDeviceEdge
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def uniform_undirected_linear_device(n_qubits: int, edge_labels: Mapping[int, Optional[UndirectedGraphDeviceEdge]]) -> UndirectedGraphDevice:
    """A uniform , undirected graph device whose qubits are arranged
    on a line.

    Uniformity refers to the fact that all edges of the same size have the same
    label.

    Args:
        n_qubits: The number of qubits.
        edge_labels: The labels to apply to all edges of a given size.

    Raises:
        ValueError: keys to edge_labels are not all at least 1.
    """
    if edge_labels and min(edge_labels) < 1:
        raise ValueError(f'edge sizes {tuple(edge_labels.keys())} must be at least 1.')
    device = UndirectedGraphDevice()
    for arity, label in edge_labels.items():
        edges = (devices.LineQubit.range(i, i + arity) for i in range(n_qubits - arity + 1))
        device += uniform_undirected_graph_device(edges, label)
    return device