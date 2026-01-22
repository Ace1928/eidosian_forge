from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def state_diagonalizing_gates(measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)
    total_indices = len(state.shape) - is_state_batched
    wires = Wires(range(total_indices))
    flattened_state = flatten_state(state, total_indices)
    return measurementprocess.process_state(flattened_state, wires)