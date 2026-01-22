import itertools
from typing import Dict, Iterator, List, Optional, Sequence, cast
import numpy as np
def get_bitstring_from_index(index: int, qubit_num: int) -> str:
    """
    Returns the bitstring in lexical order that corresponds to the given index in 0 to 2^(qubit_num)
    :param int index:
    :param int qubit_num:
    :return: the bitstring
    :rtype: str
    """
    if index > 2 ** qubit_num - 1:
        raise IndexError('Index {} too large for {} qubits.'.format(index, qubit_num))
    return bin(index)[2:].rjust(qubit_num, '0')