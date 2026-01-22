import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def with_operations(self, *contents: 'cirq.OP_TREE') -> 'cirq.Moment':
    """Returns a new moment with the given contents added.

        Args:
            *contents: New operations to add to this moment.

        Returns:
            The new moment.

        Raises:
            ValueError: If the contents given overlaps a current operation in the moment.
        """
    flattened_contents = tuple(op_tree.flatten_to_ops(contents))
    if not flattened_contents:
        return self
    m = Moment(_flatten_contents=False)
    m._qubit_to_op = self._qubit_to_op.copy()
    qubits = set(self._qubits)
    for op in flattened_contents:
        if any((q in qubits for q in op.qubits)):
            raise ValueError(f'Overlapping operations: {op}')
        qubits.update(op.qubits)
        for q in op.qubits:
            m._qubit_to_op[q] = op
    m._qubits = frozenset(qubits)
    m._operations = self._operations + flattened_contents
    m._sorted_operations = None
    m._measurement_key_objs = self._measurement_key_objs_().union(set(itertools.chain(*(protocols.measurement_key_objs(op) for op in flattened_contents))))
    m._control_keys = self._control_keys_().union(set(itertools.chain(*(protocols.control_keys(op) for op in flattened_contents))))
    return m