import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def num_cnots_required(u: np.ndarray, atol: float=1e-08) -> int:
    """Returns the min number of CNOT/CZ gates required by a two-qubit unitary.

    See Proposition III.1, III.2, III.3 in Shende et al. “Recognizing Small-
    Circuit Structure in Two-Qubit Operators and Timing Hamiltonians to Compute
    Controlled-Not Gates”.  https://arxiv.org/abs/quant-ph/0308045

    Args:
        u: A two-qubit unitary.
        atol: The absolute tolerance used to make this judgement.

    Returns:
        The number of CNOT or CZ gates required to implement the unitary.

    Raises:
        ValueError: If the shape of `u` is not 4 by 4.
    """
    if u.shape != (4, 4):
        raise ValueError(f'Expected unitary of shape (4,4), instead got {u.shape}')
    g = _gamma(transformations.to_special(u))
    a3 = -np.trace(g)
    if np.abs(a3 - 4) < atol or np.abs(a3 + 4) < atol:
        return 0
    a2 = (a3 * a3 - np.trace(g @ g)) / 2
    if np.abs(a3) < atol and np.abs(a2 - 2) < atol:
        return 1
    if np.abs(a3.imag) < atol:
        return 2
    return 3