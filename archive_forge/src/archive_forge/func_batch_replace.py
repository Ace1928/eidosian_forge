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
def batch_replace(self, replacements: Iterable[Tuple[int, 'cirq.Operation', 'cirq.Operation']]) -> None:
    """Replaces several operations in a circuit with new operations.

        Args:
            replacements: A sequence of (moment_index, old_op, new_op) tuples
                indicating operations to be replaced in this circuit. All "old"
                operations must actually be present or the edit will fail
                (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to replace wasn't present to start with.
            IndexError: Replaced in a moment that doesn't exist.
        """
    copy = self.copy()
    for i, op, new_op in replacements:
        if op not in copy._moments[i].operations:
            raise ValueError(f"Can't replace {op} @ {i} because it doesn't exist.")
        copy._moments[i] = Moment((old_op if old_op != op else new_op for old_op in copy._moments[i].operations))
    self._moments = copy._moments
    self._mutated()