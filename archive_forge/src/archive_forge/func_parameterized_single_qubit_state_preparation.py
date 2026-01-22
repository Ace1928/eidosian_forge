from typing import Sequence
import numpy as np
from pyquil.gates import MEASURE, RX, RZ
from pyquil.quil import Program
def parameterized_single_qubit_state_preparation(qubits: Sequence[int], label: str='preparation') -> Program:
    """
    Given a number of qubits, produce a program as in ``parameterized_euler_rotations`` where each
    memory region is prefixed by ``label``, where label defaults to "preparation".

    :param qubits: The number of qubits (n).
    :param label: The prefix to use when declaring memory in ``parameterized_euler_rotations``.
    :return: A parameterized ``Program`` that can be used to prepare a product state.
    """
    return parameterized_euler_rotations(qubits, prefix=label)