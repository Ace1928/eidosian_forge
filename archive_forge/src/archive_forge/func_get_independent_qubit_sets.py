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
def get_independent_qubit_sets(self) -> List[Set['cirq.Qid']]:
    """Divide circuit's qubits into independent qubit sets.

        Independent qubit sets are the qubit sets such that there are
        no entangling gates between qubits belonging to different sets.
        If this is not possible, a sequence with a single factor (the whole set of
        circuit's qubits) is returned.

        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit()
        >>> circuit.append(cirq.Moment(cirq.H(q2)))
        >>> circuit.append(cirq.Moment(cirq.CZ(q0,q1)))
        >>> circuit.append(cirq.H(q0))
        >>> print(circuit)
        0: ───────@───H───
                  │
        1: ───────@───────
        <BLANKLINE>
        2: ───H───────────
        >>> [sorted(qs) for qs in circuit.get_independent_qubit_sets()]
        [[cirq.LineQubit(0), cirq.LineQubit(1)], [cirq.LineQubit(2)]]

        Returns:
            The list of independent qubit sets.

        """
    uf = networkx.utils.UnionFind(self.all_qubits())
    for op in self.all_operations():
        if len(op.qubits) > 1:
            uf.union(*op.qubits)
    return sorted([qs for qs in uf.to_sets()], key=min)