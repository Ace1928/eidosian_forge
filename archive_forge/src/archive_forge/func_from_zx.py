from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane.transforms import transform
from pennylane.wires import Wires
def from_zx(graph, decompose_phases=True):
    """Converts a graph from `PyZX <https://pyzx.readthedocs.io/en/latest/>`_ to a PennyLane tape, if the graph is
    diagram-like.

    Args:
        graph (Graph): ZX graph in PyZX.
        decompose_phases (bool): If True the phases are decomposed, meaning that :func:`qml.RZ` and :func:`qml.RX` are
            simplified into other gates (e.g. :func:`qml.T`, :func:`qml.S`, ...).

    **Example**

    From the example for the :func:`~.to_zx` function, one can convert back the PyZX graph to a PennyLane by using the
    function :func:`~.from_zx`.

    .. code-block:: python

        import pyzx
        dev = qml.device('default.qubit', wires=2)

        @qml.transforms.to_zx
        def circuit(p):
            qml.RZ(p[0], wires=0),
            qml.RZ(p[1], wires=0),
            qml.RX(p[2], wires=1),
            qml.Z(1),
            qml.RZ(p[3], wires=0),
            qml.X(0),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[0, 1]),
            qml.SWAP(wires=[1, 0]),
            return qml.expval(qml.Z(0) @ qml.Z(1))

        params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
        g = circuit(params)

        pennylane_tape = qml.transforms.from_zx(g)

    You can check that the operations are similar but some were decomposed in the process.

    >>> pennylane_tape.operations
    [Z(0),
     T(wires=[0]),
     RX(0.1, wires=[1]),
     Z(0),
     Adjoint(T(wires=[0])),
     Z(1),
     RZ(0.3, wires=[0]),
     X(0),
     CNOT(wires=[1, 0]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[1, 0]),
     CNOT(wires=[0, 1]),
     CNOT(wires=[1, 0])]

    .. warning::

        Be careful because not all graphs are circuit-like, so the process might not be successful
        after you apply some optimization on your PyZX graph. You can extract a circuit by using the dedicated
        PyZX function.

    .. note::

        It is a PennyLane adapted and reworked `graph_to_circuit <https://github.com/Quantomatic/pyzx/blob/master/pyzx/circuit/graphparser.py>`_
        function.

        Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

    """
    operations = []
    qubits = graph.qubits()
    graph_rows = graph.rows()
    types = graph.types()
    params = graph.phases()
    rows = {}
    inputs = graph.inputs()
    for vertex in graph.vertices():
        if vertex in inputs:
            continue
        row_index = graph.row(vertex)
        if row_index in rows:
            rows[row_index].append(vertex)
        else:
            rows[row_index] = [vertex]
    for row_key in sorted(rows.keys()):
        for vertex in rows[row_key]:
            qubit_1 = qubits[vertex]
            param = params[vertex]
            type_1 = types[vertex]
            neighbors = [w for w in graph.neighbors(vertex) if graph_rows[w] < row_key]
            if len(neighbors) != 1:
                raise qml.QuantumFunctionError("Graph doesn't seem circuit like: multiple parents. Try to use the PyZX function `extract_circuit`.")
            neighbor_0 = neighbors[0]
            if qubits[neighbor_0] != qubit_1:
                raise qml.QuantumFunctionError('Cross qubit connections, the graph is not circuit-like.')
            if graph.edge_type(graph.edge(neighbor_0, vertex)) == EdgeType.HADAMARD:
                operations.append(qml.Hadamard(wires=qubit_1))
            if type_1 == VertexType.BOUNDARY:
                continue
            operations.extend(_add_one_qubit_gate(param, type_1, qubit_1, decompose_phases))
            neighbors = [w for w in graph.neighbors(vertex) if graph_rows[w] == row_key and w < vertex]
            for neighbor in neighbors:
                type_2 = types[neighbor]
                qubit_2 = qubits[neighbor]
                operations.extend(_add_two_qubit_gates(graph, vertex, neighbor, type_1, type_2, qubit_1, qubit_2))
    return QuantumScript(operations)