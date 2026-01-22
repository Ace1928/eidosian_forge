from typing import TYPE_CHECKING, cast, FrozenSet, Iterable, Mapping, Optional, Tuple
import networkx as nx
from cirq import value
from cirq.devices import device
@property
def qubit_pairs(self) -> FrozenSet[FrozenSet['cirq.GridQubit']]:
    """Returns the set of all couple-able qubits on the device.

        Each element in the outer frozenset is a 2-element frozenset representing a bidirectional
        pair.
        """
    return self._qubit_pairs