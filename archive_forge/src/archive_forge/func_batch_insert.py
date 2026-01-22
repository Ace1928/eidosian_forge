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
def batch_insert(self, insertions: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
    """Applies a batched insert operation to the circuit.

        Transparently handles the fact that earlier insertions may shift
        the index that later insertions should occur at. For example, if you
        insert an operation at index 2 and at index 4, but the insert at index 2
        causes a new moment to be created, then the insert at "4" will actually
        occur at index 5 to account for the shift from the new moment.

        All insertions are done with the strategy `cirq.InsertStrategy.EARLIEST`.

        When multiple inserts occur at the same index, the gates from the later
        inserts end up before the gates from the earlier inserts (exactly as if
        you'd called list.insert several times with the same index: the later
        inserts shift the earliest inserts forward).

        Args:
            insertions: A sequence of (insert_index, operations) pairs
                indicating operations to add into the circuit at specific
                places.
        """
    copy = self.copy()
    shift = 0
    insertions = sorted(insertions, key=lambda e: e[0])
    groups = _group_until_different(insertions, key=lambda e: e[0], val=lambda e: e[1])
    for i, group in groups:
        insert_index = i + shift
        next_index = copy.insert(insert_index, reversed(group), InsertStrategy.EARLIEST)
        if next_index > insert_index:
            shift += next_index - insert_index
    self._moments = copy._moments
    self._mutated()