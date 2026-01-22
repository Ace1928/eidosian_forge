from typing import Callable
from string import ascii_letters as alphabet
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from .utils import (
from .apply_operation import apply_operation
def calculate_expval(measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Measure the expectation value of an observable by finding the trace of obs@rho.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state.
        state (TensorLike): the state to measure.
        is_state_batched (bool): whether the state is batched or not.

    Returns:
        TensorLike: expectation value of observable wrt the state.
    """
    obs = measurementprocess.obs
    rho_mult_obs = apply_observable_einsum(obs, state, is_state_batched)
    num_wires = get_num_wires(state, is_state_batched)
    rho_mult_obs_reshaped = reshape_state_as_matrix(rho_mult_obs, num_wires)
    if is_state_batched:
        return math.real(math.stack([math.sum(math.diagonal(dm)) for dm in rho_mult_obs_reshaped]))
    return math.real(math.sum(math.diagonal(rho_mult_obs_reshaped)))