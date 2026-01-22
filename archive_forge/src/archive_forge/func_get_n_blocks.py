import warnings
from typing import Callable
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
@staticmethod
def get_n_blocks(wires, n_block_wires):
    """Returns the expected number of blocks for a set of wires and number of wires per block.
        Args:
            wires (Sequence): number of wires the template acts on
            n_block_wires (int): number of wires per block
        Returns:
            n_blocks (int): number of blocks; expected length of the template_weights argument
        """
    n_wires = len(wires)
    if not np.log2(n_wires / n_block_wires).is_integer():
        warnings.warn(f'The number of wires should be n_block_wires times 2^n; got n_wires/n_block_wires = {n_wires / n_block_wires}')
    if n_block_wires > n_wires:
        raise ValueError(f'n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}')
    n_blocks = 2 ** (np.floor(np.log2(n_wires / n_block_wires)) + 2) - 3
    return int(n_blocks)