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
def next_moments_operating_on(self, qubits: Iterable['cirq.Qid'], start_moment_index: int=0) -> Dict['cirq.Qid', int]:
    """Finds the index of the next moment that touches each qubit.

        Args:
            qubits: The qubits to find the next moments acting on.
            start_moment_index: The starting point of the search.

        Returns:
            The index of the next moment that touches each qubit. If there
            is no such moment, the next moment is specified as the number of
            moments in the circuit. Equivalently, can be characterized as one
            plus the index of the last moment after start_moment_index
            (inclusive) that does *not* act on a given qubit.
        """
    next_moments = {}
    for q in qubits:
        next_moment = self.next_moment_operating_on([q], start_moment_index)
        next_moments[q] = len(self.moments) if next_moment is None else next_moment
    return next_moments