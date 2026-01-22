from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np
from cirq import ops, protocols, value
from cirq._compat import proper_repr
def pauli_error_from_t1(t_ns: float, t1_ns: float) -> float:
    """Calculates the pauli error from T1 decay constant.

    This computes error for a specific duration, `t`.

    Args:
        t_ns: The duration of the gate in ns.
        t1_ns: The T1 decay constant in ns.

    Returns:
        Calculated Pauli error resulting from T1 decay.
    """
    t2 = 2 * t1_ns
    return (1 - np.exp(-t_ns / t2)) / 2 + (1 - np.exp(-t_ns / t1_ns)) / 4