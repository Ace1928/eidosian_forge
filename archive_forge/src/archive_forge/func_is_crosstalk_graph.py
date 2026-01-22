import abc
import itertools
from typing import Iterable, Optional, TYPE_CHECKING, Tuple, cast
from cirq import devices, ops, value
from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph
def is_crosstalk_graph(graph: UndirectedHypergraph) -> bool:
    if not isinstance(graph, UndirectedHypergraph):
        return False
    for vertex in graph.vertices:
        if not isinstance(vertex, frozenset):
            return False
        if not all((isinstance(v, ops.Qid) for v in vertex)):
            return False
    for edge, label in graph.labelled_edges.items():
        if len(edge) < 2:
            return False
        if not (label is None or callable(label)):
            return False
    return True