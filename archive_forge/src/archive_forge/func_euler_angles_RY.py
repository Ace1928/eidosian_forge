import itertools
from typing import Dict, List, Tuple, cast
import numpy as np
from pyquil.paulis import PauliTerm
def euler_angles_RY(theta: float) -> Tuple[float, float, float]:
    """
    A tuple of angles which corresponds to a ZXZXZ-decomposed ``RY`` gate.

    :param theta: The angle parameter for the ``RY`` gate.
    :return: The corresponding Euler angles for that gate.
    """
    return (0.0, theta, 0.0)