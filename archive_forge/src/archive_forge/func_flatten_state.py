from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def flatten_state(state, num_wires):
    """
    Given a non-flat, potentially batched state, flatten it.

    Args:
        state (TensorLike): A state that needs flattening
        num_wires (int): The number of wires the state represents

    Returns:
        A flat state, with an extra batch dimension if necessary
    """
    dim = 2 ** num_wires
    batch_size = math.get_batch_size(state, (2,) * num_wires, dim)
    shape = (batch_size, dim) if batch_size is not None else (dim,)
    return math.reshape(state, shape)