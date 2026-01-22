from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def pauli_error_to_decay_constant(pauli_error: float, num_qubits: int=1) -> float:
    """Calculates depolarization decay constant from pauli error.

    Args:
        pauli_error: The pauli error.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2 ** num_qubits
    return 1 - pauli_error / (1 - 1 / N / N)