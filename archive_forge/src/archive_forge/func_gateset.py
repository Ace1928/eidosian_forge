from typing import TYPE_CHECKING, cast, FrozenSet, Iterable, Mapping, Optional, Tuple
import networkx as nx
from cirq import value
from cirq.devices import device
@property
def gateset(self) -> 'cirq.Gateset':
    """Returns the `cirq.Gateset` of supported gates on this device."""
    return self._gateset