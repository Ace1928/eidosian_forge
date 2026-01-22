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
def earliest_available_moment(self, op: 'cirq.Operation', *, end_moment_index: Optional[int]=None) -> int:
    """Finds the index of the earliest (i.e. left most) moment which can accommodate `op`.

        Note that, unlike `circuit.prev_moment_operating_on`, this method also takes care of
        implicit dependencies between measurements and classically controlled operations (CCO)
        that depend on the results of those measurements. Therefore, using this method, a CCO
        `op` would not be allowed to move left past a measurement it depends upon.

        Args:
            op: Operation for which the earliest moment that can accommodate it needs to be found.
            end_moment_index: The moment index just after the starting point of the reverse search.
                Defaults to the length of the list of moments.

        Returns:
            Index of the earliest matching moment. Returns `end_moment_index` if no moment on left
            is available.
        """
    if end_moment_index is None:
        end_moment_index = len(self.moments)
    last_available = end_moment_index
    k = end_moment_index
    op_control_keys = protocols.control_keys(op)
    op_measurement_keys = protocols.measurement_key_objs(op)
    op_qubits = op.qubits
    while k > 0:
        k -= 1
        moment = self._moments[k]
        if moment.operates_on(op_qubits):
            return last_available
        moment_measurement_keys = moment._measurement_key_objs_()
        if not op_measurement_keys.isdisjoint(moment_measurement_keys) or not op_control_keys.isdisjoint(moment_measurement_keys) or (not moment._control_keys_().isdisjoint(op_measurement_keys)):
            return last_available
        if self._can_add_op_at(k, op):
            last_available = k
    return last_available