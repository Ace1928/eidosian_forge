import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def operates_on(self, qubits: Iterable['cirq.Qid']) -> bool:
    """Determines if the moment has operations touching the given qubits.

        Args:
            qubits: The qubits that may or may not be touched by operations.

        Returns:
            Whether this moment has operations involving the qubits.
        """
    return not self._qubits.isdisjoint(qubits)