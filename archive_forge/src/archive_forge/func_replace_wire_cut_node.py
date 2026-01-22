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
def replace_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """
    Replace a :class:`~.WireCut` node in the graph with a :class:`~.MeasureNode`
    and :class:`~.PrepareNode`.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        node (WireCut): the  :class:`~.WireCut` node to be replaced with a :class:`~.MeasureNode`
            and :class:`~.PrepareNode`
        graph (nx.MultiDiGraph): the graph containing the node to be replaced

    **Example**

    Consider the following circuit with a manually-placed wire cut:

    .. code-block:: python

        wire_cut = qml.WireCut(wires=0)

        ops = [
            qml.RX(0.4, wires=0),
            wire_cut,
            qml.RY(0.5, wires=0),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can find the circuit graph and remove the wire cut node using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_node(wire_cut, graph)
    """
    node_obj = WrappedObj(node)
    predecessors = graph.pred[node_obj]
    successors = graph.succ[node_obj]
    predecessor_on_wire = {}
    for op, data in predecessors.items():
        for d in data.values():
            wire = d['wire']
            predecessor_on_wire[wire] = op
    successor_on_wire = {}
    for op, data in successors.items():
        for d in data.values():
            wire = d['wire']
            successor_on_wire[wire] = op
    order = graph.nodes[node_obj]['order']
    graph.remove_node(node_obj)
    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)
        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)
        meas_node = WrappedObj(meas)
        prep_node = WrappedObj(prep)
        graph.add_node(meas_node, order=order)
        graph.add_node(prep_node, order=order)
        graph.add_edge(meas_node, prep_node, wire=wire)
        if predecessor is not None:
            graph.add_edge(predecessor, meas_node, wire=wire)
        if successor is not None:
            graph.add_edge(prep_node, successor, wire=wire)