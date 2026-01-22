from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.utils import (
def negativity(state, qargs):
    """Calculates the negativity

    The mathematical expression for negativity is given by:
    .. math::
        {\\cal{N}}(\\rho) = \\frac{|| \\rho^{T_A}|| - 1 }{2}

    Args:
        state (Statevector or DensityMatrix): a quantum state.
        qargs (list): The subsystems to be transposed.

    Returns:
        negv (float): Negativity value of the quantum state

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
    """
    if isinstance(state, Statevector):
        state = DensityMatrix(state)
    state = state.partial_transpose(qargs)
    singular_values = np.linalg.svd(state.data, compute_uv=False)
    eigvals = np.sum(singular_values)
    negv = (eigvals - 1) / 2
    return negv