from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def synth_cx_cz_depth_line_my(mat_x: np.ndarray, mat_z: np.ndarray) -> QuantumCircuit:
    """
    Joint synthesis of a -CZ-CX- circuit for linear nearest neighbour (LNN) connectivity,
    with 2-qubit depth at most 5n, based on Maslov and Yang.
    This method computes the CZ circuit inside the CX circuit via phase gate insertions.

    Args:
        mat_z : a boolean symmetric matrix representing a CZ circuit.
            ``mat_z[i][j]=1`` represents a ``cz(i,j)`` gate

        mat_x : a boolean invertible matrix representing a CX circuit.

    Returns:
        A circuit implementation of a CX circuit following a CZ circuit,
        denoted as a -CZ-CX- circuit,in two-qubit depth at most ``5n``, for LNN connectivity.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
        2. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary
           Hadamard-free Clifford transformations they generate*,
           `arXiv:2210.16195 <https://arxiv.org/abs/2210.16195>`_.
    """
    n = len(mat_x)
    mat_x = calc_inverse_matrix(mat_x)
    cx_instructions_rows_m2nw, cx_instructions_rows_nw2id = _optimize_cx_circ_depth_5n_line(mat_x)
    phase_schedule = _initialize_phase_schedule(mat_z)
    seq = _make_seq(n)
    swap_plus = _swap_plus(cx_instructions_rows_nw2id, seq)
    _update_phase_schedule(n, phase_schedule, swap_plus)
    qc = _apply_phase_to_nw_circuit(n, phase_schedule, seq, swap_plus)
    for i, j in reversed(cx_instructions_rows_m2nw):
        qc.cx(i, j)
    return qc