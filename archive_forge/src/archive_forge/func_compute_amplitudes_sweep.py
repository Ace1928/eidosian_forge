import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def compute_amplitudes_sweep(self, program: 'cirq.AbstractCircuit', bitstrings: Sequence[int], params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Sequence[Sequence[complex]]:
    """Wraps computed amplitudes in a list.

        Prefer overriding `compute_amplitudes_sweep_iter`.
        """
    return list(self.compute_amplitudes_sweep_iter(program, bitstrings, params, qubit_order))