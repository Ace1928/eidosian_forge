import cmath
from typing import Tuple
import numpy as np
def phase_flip_operators(p: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the phase flip kraus operators
    """
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p) * Z
    return (k0, k1)