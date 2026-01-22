import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def mapped_circuit(self, deep: bool=False) -> 'cirq.Circuit':
    """Applies all maps to the contained circuit and returns the result.

        Args:
            deep: If true, this will also call mapped_circuit on any
                CircuitOperations this object contains.

        Returns:
            The contained circuit with all other member variables (repetitions,
            qubit mapping, parameterization, etc.) applied to it. This behaves
            like `cirq.decompose(self)`, but preserving moment structure.
        """
    self._ensure_deterministic_loop_count()
    if self.repetitions == 0:
        return circuits.Circuit()
    circuit = circuits.Circuit((self._mapped_single_loop(rep) for rep in self.repetition_ids)) if self.repetition_ids is not None and self.use_repetition_ids and protocols.is_measurement(self.circuit) else self._mapped_single_loop() * cast(IntParam, abs(self.repetitions))
    if deep:
        circuit = circuit.map_operations(lambda op: op.mapped_circuit(deep=True) if isinstance(op, CircuitOperation) else op)
    return circuit