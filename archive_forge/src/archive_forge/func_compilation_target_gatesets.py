from typing import TYPE_CHECKING, cast, FrozenSet, Iterable, Mapping, Optional, Tuple
import networkx as nx
from cirq import value
from cirq.devices import device
@property
def compilation_target_gatesets(self) -> Tuple['cirq.CompilationTargetGateset', ...]:
    """Returns a sequence of valid `cirq.CompilationTargetGateset`s for this device."""
    return self._compilation_target_gatesets