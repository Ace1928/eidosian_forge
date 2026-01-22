import abc
import collections
from typing import (
import numpy as np
from cirq import devices, ops, protocols, study, value
from cirq.sim.simulation_product_state import SimulationProductState
from cirq.sim.simulation_state import TSimulationState
from cirq.sim.simulation_state_base import SimulationStateBase
from cirq.sim.simulator import (
def get_state_containing_qubit(self, qubit: 'cirq.Qid') -> TSimulationState:
    """Returns the independent state space containing the qubit.

        Args:
            qubit: The qubit whose state space is required.

        Returns:
            The state space containing the qubit."""
    return self._final_simulator_state[qubit]