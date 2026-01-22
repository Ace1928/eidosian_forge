import warnings
from typing import Callable
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def compute_indices(wires, n_block_wires):
    """Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block

    Returns:
        layers (array): array of wire labels for each block
    """
    n_wires = len(wires)
    if n_block_wires % 2 != 0:
        raise ValueError(f'n_block_wires must be an even integer; got {n_block_wires}')
    if n_block_wires < 2:
        raise ValueError(f'number of wires in each block must be larger than or equal to 2; got n_block_wires = {n_block_wires}')
    if n_block_wires > n_wires:
        raise ValueError(f'n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}')
    if not np.log2(n_wires / n_block_wires).is_integer():
        warnings.warn(f'The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires / n_block_wires}')
    n_layers = np.floor(np.log2(n_wires / n_block_wires)).astype(int) * 2 + 1
    wires_list = []
    wires_list.append(list(wires[0:n_block_wires]))
    highest_index = n_block_wires
    for i in range(n_layers - 1):
        n_elements_pre = 2 ** ((i + 1) // 2)
        if i % 2 == 0:
            new_list = []
            list_len = len(wires_list)
            for j in range(list_len - n_elements_pre, list_len):
                new_wires = [wires[k] for k in range(highest_index, highest_index + n_block_wires // 2)]
                highest_index += n_block_wires // 2
                new_list.append(wires_list[j][0:n_block_wires // 2] + new_wires)
                new_wires = [wires[k] for k in range(highest_index, highest_index + n_block_wires // 2)]
                highest_index += n_block_wires // 2
                new_list.append(new_wires + wires_list[j][n_block_wires // 2:])
            wires_list = wires_list + new_list
        else:
            list_len = len(wires_list)
            new_list = []
            for j in range(list_len - n_elements_pre, list_len - 1):
                new_list.append(wires_list[j][n_block_wires // 2:] + wires_list[j + 1][0:n_block_wires // 2])
            new_list.append(wires_list[j + 1][n_block_wires // 2:] + wires_list[list_len - n_elements_pre][0:n_block_wires // 2])
            wires_list = wires_list + new_list
    return tuple((tuple(l) for l in wires_list[::-1]))