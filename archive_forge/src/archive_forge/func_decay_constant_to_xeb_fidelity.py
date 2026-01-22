from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def decay_constant_to_xeb_fidelity(decay_constant: float, num_qubits: int=2) -> float:
    """Calculates the XEB fidelity from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated XEB fidelity.
    """
    N = 2 ** num_qubits
    return 1 - (1 - decay_constant) * (1 - 1 / N)