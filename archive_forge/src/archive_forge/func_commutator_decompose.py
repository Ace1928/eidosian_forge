from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def commutator_decompose(u_so3: np.ndarray, check_input: bool=True) -> tuple[GateSequence, GateSequence]:
    """Decompose an :math:`SO(3)`-matrix, :math:`U` as a balanced commutator.

    This function finds two :math:`SO(3)` matrices :math:`V, W` such that the input matrix
    equals

    .. math::

        U = V^\\dagger W^\\dagger V W.

    For this decomposition, the following statement holds


    .. math::

        ||V - I||_F, ||W - I||_F \\leq \\frac{\\sqrt{||U - I||_F}}{2},

    where :math:`I` is the identity and :math:`||\\cdot ||_F` is the Frobenius norm.

    Args:
        u_so3: SO(3)-matrix that needs to be decomposed as balanced commutator.
        check_input: If True, checks whether the input matrix is actually SO(3).

    Returns:
        Tuple of GateSequences from SO(3)-matrices :math:`V, W`.

    Raises:
        ValueError: if ``u_so3`` is not an SO(3)-matrix.
    """
    if check_input:
        _check_is_so3(u_so3)
        if not is_identity_matrix(u_so3.dot(u_so3.T)):
            raise ValueError('Input matrix is not orthogonal.')
    angle = _solve_decomposition_angle(u_so3)
    vx = _compute_rotation_from_angle_and_axis(angle, np.array([1, 0, 0]))
    wy = _compute_rotation_from_angle_and_axis(angle, np.array([0, 1, 0]))
    commutator = _compute_commutator_so3(vx, wy)
    u_so3_axis = _compute_rotation_axis(u_so3)
    commutator_axis = _compute_rotation_axis(commutator)
    sim_matrix = _compute_rotation_between(commutator_axis, u_so3_axis)
    sim_matrix_dagger = np.conj(sim_matrix).T
    v = np.dot(np.dot(sim_matrix, vx), sim_matrix_dagger)
    w = np.dot(np.dot(sim_matrix, wy), sim_matrix_dagger)
    return (GateSequence.from_matrix(v), GateSequence.from_matrix(w))