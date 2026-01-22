import itertools
from typing import Iterable, Tuple, Dict
import networkx as nx
import cirq
def get_linear_device_graph(n_qubits: int) -> nx.Graph:
    """Gets the graph of a linearly connected device."""
    qubits = cirq.LineQubit.range(n_qubits)
    edges = [tuple(qubits[i:i + 2]) for i in range(n_qubits - 1)]
    return nx.Graph(edges)