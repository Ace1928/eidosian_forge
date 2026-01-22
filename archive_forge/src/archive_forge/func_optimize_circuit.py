import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def optimize_circuit(self, circuit: 'cirq.Circuit') -> None:
    circuit._moments = [*transformers.expand_composite(circuit, no_decomp=self.no_decomp)]