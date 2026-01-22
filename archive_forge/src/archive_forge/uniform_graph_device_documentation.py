from typing import Any, Dict, Hashable, Iterable, Mapping, Optional
from cirq import devices, ops
from cirq.contrib.graph_device.graph_device import UndirectedGraphDevice, UndirectedGraphDeviceEdge
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
A uniform , undirected graph device whose qubits are arranged
    on a line.

    Uniformity refers to the fact that all edges of the same size have the same
    label.

    Args:
        n_qubits: The number of qubits.
        edge_labels: The labels to apply to all edges of a given size.

    Raises:
        ValueError: keys to edge_labels are not all at least 1.
    