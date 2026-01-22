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
def insert_at_frontier(self, operations: 'cirq.OP_TREE', start: int, frontier: Optional[Dict['cirq.Qid', int]]=None) -> Dict['cirq.Qid', int]:
    """Inserts operations inline at frontier.

        Args:
            operations: The operations to insert.
            start: The moment at which to start inserting the operations.
            frontier: frontier[q] is the earliest moment in which an operation
                acting on qubit q can be placed.

        Raises:
            ValueError: If the frontier given is after start.
        """
    if frontier is None:
        frontier = defaultdict(lambda: 0)
    flat_ops = tuple(ops.flatten_to_ops(operations))
    if not flat_ops:
        return frontier
    qubits = set((q for op in flat_ops for q in op.qubits))
    if any((frontier[q] > start for q in qubits)):
        raise ValueError('The frontier for qubits on which the operationsto insert act cannot be after start.')
    next_moments = self.next_moments_operating_on(qubits, start)
    insertion_indices, _ = _pick_inserted_ops_moment_indices(flat_ops, start, frontier)
    self._push_frontier(frontier, next_moments)
    self._insert_operations(flat_ops, insertion_indices)
    return frontier