import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
def check_all_resolved(circuit):
    """Raises if the circuit contains unresolved symbols."""
    if protocols.is_parameterized(circuit):
        unresolved = [op for moment in circuit for op in moment if protocols.is_parameterized(op)]
        raise ValueError(f'Circuit contains ops whose symbols were not specified in parameter sweep. Ops: {unresolved}')