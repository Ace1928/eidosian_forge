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
def expand_fragment_tape(tape: QuantumTape) -> Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]:
    """
    Expands a fragment tape into a sequence of tapes for each configuration of the contained
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): the fragment tape containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded

    Returns:
        Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]: the
        tapes corresponding to each configuration and the order of preparation nodes and
        measurement nodes used in the expansion

    **Example**

    Consider the following circuit, which contains a :class:`~.MeasureNode` and
    :class:`~.PrepareNode` operation:

    .. code-block:: python

        ops = [
            qml.qcut.PrepareNode(wires=0),
            qml.RX(0.5, wires=0),
            qml.qcut.MeasureNode(wires=0),
        ]
        tape = qml.tape.QuantumTape(ops)

    We can expand over the measurement and preparation nodes using:

    >>> tapes, prep, meas = qml.qcut.expand_fragment_tape(tape)
    >>> for t in tapes:
    ...     print(qml.drawer.tape_text(t, decimals=1))
    0: ──I──RX(0.5)─┤  <I>  <Z>
    0: ──I──RX(0.5)─┤  <X>
    0: ──I──RX(0.5)─┤  <Y>
    0: ──X──RX(0.5)─┤  <I>  <Z>
    0: ──X──RX(0.5)─┤  <X>
    0: ──X──RX(0.5)─┤  <Y>
    0: ──H──RX(0.5)─┤  <I>  <Z>
    0: ──H──RX(0.5)─┤  <X>
    0: ──H──RX(0.5)─┤  <Y>
    0: ──H──S──RX(0.5)─┤  <I>  <Z>
    0: ──H──S──RX(0.5)─┤  <X>
    0: ──H──S──RX(0.5)─┤  <Y>
    """
    prepare_nodes = [o for o in tape.operations if isinstance(o, PrepareNode)]
    measure_nodes = [o for o in tape.operations if isinstance(o, MeasureNode)]
    wire_map = {mn.wires[0]: i for i, mn in enumerate(measure_nodes)}
    n_meas = len(measure_nodes)
    if n_meas >= 1:
        measure_combinations = qml.pauli.partition_pauli_group(len(measure_nodes))
    else:
        measure_combinations = [['']]
    tapes = []
    for prepare_settings in product(range(len(PREPARE_SETTINGS)), repeat=len(prepare_nodes)):
        for measure_group in measure_combinations:
            if n_meas >= 1:
                group = [string_to_pauli_word(paulis, wire_map=wire_map) for paulis in measure_group]
            else:
                group = []
            prepare_mapping = {id(n): PREPARE_SETTINGS[s] for n, s in zip(prepare_nodes, prepare_settings)}
            ops_list = []
            with qml.QueuingManager.stop_recording():
                for op in tape.operations:
                    if isinstance(op, PrepareNode):
                        w = op.wires[0]
                        ops_list.extend(prepare_mapping[id(op)](w))
                    elif not isinstance(op, MeasureNode):
                        ops_list.append(op)
                measurements = _get_measurements(group, tape.measurements)
            qs = qml.tape.QuantumScript(ops=ops_list, measurements=measurements)
            tapes.append(qs)
    return (tapes, prepare_nodes, measure_nodes)