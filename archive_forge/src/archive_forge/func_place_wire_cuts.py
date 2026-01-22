import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation
from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy
def place_wire_cuts(graph: MultiDiGraph, cut_edges: Sequence[Tuple[Operation, Operation, Any]]) -> MultiDiGraph:
    """Inserts a :class:`~.WireCut` node for each provided cut edge into a circuit graph.

    Args:
        graph (nx.MultiDiGraph): The original (tape-converted) graph to be cut.
        cut_edges (Sequence[Tuple[Operation, Operation, Any]]): List of ``MultiDiGraph`` edges
            to be replaced with a :class:`~.WireCut` node. Each 3-tuple represents the source node, the
            target node, and the wire key of the (multi)edge.

    Returns:
        MultiDiGraph: Copy of the input graph with :class:`~.WireCut` nodes inserted.

    **Example**

    Consider the following 2-wire circuit with one CNOT gate connecting the wires:

    .. code-block:: python

        ops = [
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
            qml.CNOT(wires=[0, "a"]),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape))
     0: ──RX(0.432)──╭●──┤ ⟨Z⟩
     a: ──RY(0.543)──╰X──┤

    If we know we want to place a :class:`~.WireCut` node between the nodes corresponding to the
    ``RY(0.543, wires=["a"])`` and ``CNOT(wires=[0, 'a'])`` operations after the tape is constructed,
    we can first find the edge in the graph:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> op0, op1 = tape.operations[1], tape.operations[2]
    >>> cut_edges = [e for e in graph.edges if e[0] is op0 and e[1] is op1]
    >>> cut_edges
    [(RY(0.543, wires=['a']), CNOT(wires=[0, 'a']), 0)]

    Then feed it to this function for placement:

    >>> cut_graph = qml.qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
    >>> cut_graph
    <networkx.classes.multidigraph.MultiDiGraph at 0x7f7251ac1220>

    And visualize the cut by converting back to a tape:

    >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX(0.432)──────╭●──┤ ⟨Z⟩
     a: ──RY(0.543)──//──╰X──┤
    """
    cut_graph = graph.copy()
    for op0, op1, wire_key in cut_edges:
        order = cut_graph.nodes[op0]['order'] + 1
        wire = cut_graph.edges[op0, op1, wire_key]['wire']
        cut_graph.remove_edge(op0, op1, wire_key)
        for op, o in cut_graph.nodes(data='order'):
            if o >= order:
                cut_graph.nodes[op]['order'] += 1
        wire_cut = WireCut(wires=wire)
        wire_cut_node = WrappedObj(wire_cut)
        cut_graph.add_node(wire_cut_node, order=order)
        cut_graph.add_edge(op0, wire_cut_node, wire=wire)
        cut_graph.add_edge(wire_cut_node, op1, wire=wire)
    return cut_graph