import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
def transpose_to_qubit_order(self, qubits: Sequence['cirq.Qid'], *, inplace=False) -> Self:
    """Physically reindexes the state by the new basis.

        Args:
            qubits: The desired qubit order.
            inplace: True to perform this operation inplace.

        Returns:
            The state with qubit order transposed and underlying representation
            updated.

        Raises:
            ValueError: If the provided qubits do not match the existing ones.
        """
    if len(self.qubits) != len(qubits) or set(qubits) != set(self.qubits):
        raise ValueError(f'Qubits do not match. Existing: {self.qubits}, provided: {qubits}')
    args = self if inplace else copy.copy(self)
    args._state = self._state.reindex(self.get_axes(qubits))
    args._set_qubits(qubits)
    return args