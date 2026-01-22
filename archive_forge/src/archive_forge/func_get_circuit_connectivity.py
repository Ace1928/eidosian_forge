import operator
from typing import Callable, Iterable, List, TYPE_CHECKING
import re
import networkx as nx
from cirq import circuits, ops
import cirq.contrib.acquaintance as cca
from cirq.contrib.circuitdag import CircuitDag
from cirq.contrib.routing.swap_network import SwapNetwork
def get_circuit_connectivity(circuit: 'cirq.Circuit') -> nx.Graph:
    """Return a graph of all 2q interactions in a circuit.

    Nodes are qubits and undirected edges correspond to any two-qubit
    operation.
    """
    g = nx.Graph()
    for op in circuit.all_operations():
        n_qubits = len(op.qubits)
        if n_qubits > 2:
            raise ValueError(f'Cannot build a graph out of a circuit that contains {n_qubits}-qubit operations')
        if n_qubits == 2:
            g.add_edge(*op.qubits)
    return g