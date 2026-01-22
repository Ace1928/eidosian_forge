from typing import Any, Iterable, Mapping
import networkx as nx
import cirq
from cirq_aqt import aqt_target_gateset
@property
def twoq_gates_duration(self) -> 'cirq.DURATION_LIKE':
    """Return the maximum duration of an operation on two-qubit gates."""
    return self._twoq_gates_duration