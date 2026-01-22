from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def xeb_fidelity_to_decay_constant(xeb_fidelity: float, num_qubits: int=2) -> float:
    """Calculates the depolarization decay constant from XEB fidelity.

    Args:
        xeb_fidelity: The XEB fidelity.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2 ** num_qubits
    return 1 - (1 - xeb_fidelity) / (1 - 1 / N)