import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def get_device_edge_from_op(self, operation: ops.Operation) -> UndirectedGraphDeviceEdge:
    return self.device_graph.labelled_edges[frozenset(operation.qubits)]