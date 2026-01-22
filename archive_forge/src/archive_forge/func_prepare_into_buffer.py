from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, protocols, qis, sim
from cirq._compat import proper_repr
from cirq.linalg import transformations
from cirq.sim.simulation_state import SimulationState, strat_act_on_from_apply_decompose
def prepare_into_buffer(k: int):
    linalg.targeted_left_multiply(left_matrix=kraus_tensors[k], right_target=self._state_vector, target_axes=axes, out=self._buffer)