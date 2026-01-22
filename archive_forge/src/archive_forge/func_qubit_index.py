from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def qubit_index(qubit: QubitDesignator) -> int:
    """
    Get the index of a QubitDesignator.

    :param qubit: the qubit designator.
    :return: An int that is the qubit index.
    """
    if isinstance(qubit, Qubit):
        return qubit.index
    elif isinstance(qubit, int):
        return qubit
    else:
        raise TypeError(f'Cannot unwrap unaddressed QubitPlaceholder: {qubit}')