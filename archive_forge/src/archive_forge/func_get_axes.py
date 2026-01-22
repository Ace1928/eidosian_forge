import abc
import copy
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase
def get_axes(self, qubits: Sequence['cirq.Qid']) -> List[int]:
    return [self.qubit_map[q] for q in qubits]