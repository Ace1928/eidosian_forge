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
def batch_insert_into(self, insert_intos: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
    """Inserts operations into empty spaces in existing moments.

        If any of the insertions fails (due to colliding with an existing
        operation), this method fails without making any changes to the circuit.

        Args:
            insert_intos: A sequence of (moment_index, new_op_tree)
                pairs indicating a moment to add new operations into.

        Raises:
            ValueError: One of the insertions collided with an existing
                operation.
            IndexError: Inserted into a moment index that doesn't exist.
        """
    copy = self.copy()
    for i, insertions in insert_intos:
        copy._moments[i] = copy._moments[i].with_operations(insertions)
    self._moments = copy._moments
    self._mutated()