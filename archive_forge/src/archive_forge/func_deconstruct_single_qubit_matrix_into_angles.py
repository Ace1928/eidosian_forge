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
def deconstruct_single_qubit_matrix_into_angles(mat: np.ndarray) -> Tuple[float, float, float]:
    """Breaks down a 2x2 unitary into ZYZ angle parameters.

    Given a unitary U, this function returns three angles: $\\phi_0, \\phi_1, \\phi_2$,
    such that:  $U = Z^{\\phi_2 / \\pi} Y^{\\phi_1 / \\pi} Z^{\\phi_0/ \\pi}$
    for the Pauli matrices Y and Z.  That is, phasing around Z by $\\phi_0$ radians,
    then rotating around Y by $\\phi_1$ radians, and then phasing again by
    $\\phi_2$ radians will produce the same effect as the original unitary.
    (Note that the matrices are applied right to left.)

    Args:
        mat: The 2x2 unitary matrix to break down.

    Returns:
        A tuple containing the amount to phase around Z, then rotate around Y,
        then phase around Z (all in radians).
    """
    right_phase = cmath.phase(mat[0, 1] * np.conj(mat[0, 0])) + math.pi
    mat = np.dot(mat, _phase_matrix(-right_phase))
    bottom_phase = cmath.phase(mat[1, 0] * np.conj(mat[0, 0]))
    mat = np.dot(_phase_matrix(-bottom_phase), mat)
    rotation = math.atan2(abs(mat[1, 0]), abs(mat[0, 0]))
    mat = np.dot(_rotation_matrix(-rotation), mat)
    diagonal_phase = cmath.phase(mat[1, 1] * np.conj(mat[0, 0]))
    return (right_phase + diagonal_phase, rotation * 2, bottom_phase)