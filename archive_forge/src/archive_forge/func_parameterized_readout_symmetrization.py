from typing import Sequence
import numpy as np
from pyquil.gates import MEASURE, RX, RZ
from pyquil.quil import Program
def parameterized_readout_symmetrization(qubits: Sequence[int], label: str='symmetrization') -> Program:
    """
    Given a number of qubits (n), produce a ``Program`` with an ``RX`` instruction on qubits
    0 through n-1, parameterized by memory regions label[0] through label[n-1], where "label"
    defaults to "symmetrization".

    :param qubits: The number of qubits (n).
    :param label: The name of the declared memory region.
    :return: A ``Program`` with parameterized ``RX`` gates on n qubits.
    """
    p = Program()
    symmetrization = p.declare(f'{label}', 'REAL', len(qubits))
    for idx, q in enumerate(qubits):
        p += RX(symmetrization[idx], q)
    return p