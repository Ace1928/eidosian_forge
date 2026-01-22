import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires
from .utils import MeasureNode, PrepareNode
def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """
    Converts a quantum tape to a directed multigraph.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): tape to be converted into a directed multigraph

    Returns:
        nx.MultiDiGraph: a directed multigraph that captures the circuit structure
        of the input tape. The nodes of the graph are formatted as ``WrappedObj(op)``, where
        ``WrappedObj.obj`` is the operator.

    **Example**

    Consider the following tape:

    .. code-block:: python

        ops = [
            qml.RX(0.4, wires=0),
            qml.RY(0.9, wires=0),
            qml.CNOT(wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.Z(1))]
        tape = qml.tape.QuantumTape(ops,)

    Its corresponding circuit graph can be found using

    >>> qml.qcut.tape_to_graph(tape)
    <networkx.classes.multidigraph.MultiDiGraph at 0x7fe41cbd7210>
    """
    graph = MultiDiGraph()
    wire_latest_node = {w: None for w in tape.wires}
    for order, op in enumerate(tape.operations):
        _add_operator_node(graph, op, order, wire_latest_node)
    order += 1
    for m in tape.measurements:
        obs = getattr(m, 'obs', None)
        if obs is not None and isinstance(obs, Tensor):
            if isinstance(m, SampleMP):
                raise ValueError('Sampling from tensor products of observables is not supported in circuit cutting')
            for o in obs.obs:
                m_ = m.__class__(obs=o)
                _add_operator_node(graph, m_, order, wire_latest_node)
        elif isinstance(m, SampleMP) and obs is None:
            for w in m.wires:
                s_ = qml.sample(qml.Projector([1], wires=w))
                _add_operator_node(graph, s_, order, wire_latest_node)
        else:
            _add_operator_node(graph, m, order, wire_latest_node)
            order += 1
    return graph