import cmath
from typing import Tuple
import numpy as np
def depolarizing_operators(p: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the phase damping Kraus operators
    """
    k0 = np.sqrt(1.0 - p) * I
    k1 = np.sqrt(p / 3.0) * X
    k2 = np.sqrt(p / 3.0) * Y
    k3 = np.sqrt(p / 3.0) * Z
    return (k0, k1, k2, k3)