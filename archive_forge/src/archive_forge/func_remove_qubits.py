from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def remove_qubits(self, qubits: Sequence['cirq.Qid']):
    ret = super().remove_qubits(qubits)
    if ret is not NotImplemented:
        return ret
    extracted, remainder = self.factor(qubits, inplace=True)
    remainder._state._density_matrix *= extracted._state._density_matrix.reshape(-1)[0]
    return remainder