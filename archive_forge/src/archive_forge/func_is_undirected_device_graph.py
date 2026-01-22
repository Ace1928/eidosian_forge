import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def is_undirected_device_graph(graph: UndirectedHypergraph) -> bool:
    if not isinstance(graph, UndirectedHypergraph):
        return False
    if not all((isinstance(v, ops.Qid) for v in graph.vertices)):
        return False
    for _, label in graph.labelled_edges.items():
        if not (label is None or isinstance(label, UndirectedGraphDeviceEdge)):
            return False
    return True