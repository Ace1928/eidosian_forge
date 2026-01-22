from typing import TYPE_CHECKING, cast, FrozenSet, Iterable, Mapping, Optional, Tuple
import networkx as nx
from cirq import value
from cirq.devices import device
@property
def isolated_qubits(self) -> FrozenSet['cirq.GridQubit']:
    """Returns the set of all isolated qubits on the device (if applicable)."""
    return self._isolated_qubits