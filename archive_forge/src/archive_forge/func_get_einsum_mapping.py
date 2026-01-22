import functools
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np
def get_einsum_mapping(op: qml.operation.Operator, state, map_indices, is_state_batched: bool=False):
    """Finds the indices for einsum to apply kraus operators to a mixed state

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        map_indices (function): Maps the calculated indices to an einsum indices string
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        str: Indices mapping that defines the einsum
    """
    num_ch_wires = len(op.wires)
    num_wires = int((len(qml.math.shape(state)) - is_state_batched) / 2)
    rho_dim = 2 * num_wires
    state_indices = alphabet[:rho_dim]
    row_wires_list = op.wires.tolist()
    row_indices = ''.join(alphabet_array[row_wires_list].tolist())
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = ''.join(alphabet_array[col_wires_list].tolist())
    new_row_indices = alphabet[rho_dim:rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires:rho_dim + 2 * num_ch_wires]
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires:rho_dim + 2 * num_ch_wires + 1]
    return map_indices(state_indices=state_indices, kraus_index=kraus_index, row_indices=row_indices, new_row_indices=new_row_indices, col_indices=col_indices, new_col_indices=new_col_indices)