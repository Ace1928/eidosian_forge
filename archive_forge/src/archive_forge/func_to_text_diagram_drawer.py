import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def to_text_diagram_drawer(self, *, use_unicode_characters: bool=True, qubit_namer: Optional[Callable[['cirq.Qid'], str]]=None, transpose: bool=False, include_tags: bool=True, draw_moment_groups: bool=True, precision: Optional[int]=3, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, get_circuit_diagram_info: Optional[Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo']]=None) -> 'cirq.TextDiagramDrawer':
    """Returns a TextDiagramDrawer with the circuit drawn into it.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            qubit_namer: Names qubits in diagram. Defaults to using _circuit_diagram_info_ or str.
            transpose: Arranges qubit wires vertically instead of horizontally.
            include_tags: Whether to include tags in the operation.
            draw_moment_groups: Whether to draw moment symbol or not
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the diagram.
            get_circuit_diagram_info: Gets circuit diagram info. Defaults to
                protocol with fallback.

        Returns:
            The TextDiagramDrawer instance.
        """
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
    cbits = tuple(sorted(set((key for op in self.all_operations() for key in protocols.control_keys(op))), key=str))
    labels = qubits + cbits
    label_map = {labels[i]: i for i in range(len(labels))}

    def default_namer(label_entity):
        info = protocols.circuit_diagram_info(label_entity, default=None)
        qubit_name = info.wire_symbols[0] if info else str(label_entity)
        return qubit_name + ('' if transpose else ': ')
    if qubit_namer is None:
        qubit_namer = default_namer
    diagram = TextDiagramDrawer()
    diagram.write(0, 0, '')
    for label_entity, i in label_map.items():
        name = qubit_namer(label_entity) if isinstance(label_entity, ops.Qid) else default_namer(label_entity)
        diagram.write(0, i, name)
    first_annotation_row = max(label_map.values(), default=0) + 1
    if any((isinstance(op.gate, cirq.GlobalPhaseGate) for op in self.all_operations())):
        diagram.write(0, max(label_map.values(), default=0) + 1, 'global phase:')
        first_annotation_row += 1
    moment_groups: List[Tuple[int, int]] = []
    for moment in self.moments:
        _draw_moment_in_diagram(moment=moment, use_unicode_characters=use_unicode_characters, label_map=label_map, out_diagram=diagram, precision=precision, moment_groups=moment_groups, get_circuit_diagram_info=get_circuit_diagram_info, include_tags=include_tags, first_annotation_row=first_annotation_row, transpose=transpose)
    w = diagram.width()
    for i in label_map.values():
        diagram.horizontal_line(i, 0, w, doubled=not isinstance(labels[i], ops.Qid))
    if moment_groups and draw_moment_groups:
        _draw_moment_groups_in_diagram(moment_groups, use_unicode_characters, diagram)
    if transpose:
        diagram = diagram.transpose()
    return diagram