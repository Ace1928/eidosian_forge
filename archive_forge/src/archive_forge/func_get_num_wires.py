import functools
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np
def get_num_wires(state, is_state_batched: bool=False):
    """Finds the number of wires associated with a state

    Args:
        state (TensorLike): A device compatible state that may or may not be batched
        is_state_batched (int): Boolean representing whether the state is batched or not

    Returns:
        int: Number of wires associated with state
    """
    len_row_plus_col = len(math.shape(state)) - is_state_batched
    return int(len_row_plus_col / 2)