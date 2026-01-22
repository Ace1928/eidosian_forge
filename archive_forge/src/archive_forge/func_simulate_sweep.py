import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def simulate_sweep(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> List[TSimulationTrialResult]:
    """Wraps computed states in a list.

        Prefer overriding `simulate_sweep_iter`.
        """
    return list(self.simulate_sweep_iter(program, params, qubit_order, initial_state))