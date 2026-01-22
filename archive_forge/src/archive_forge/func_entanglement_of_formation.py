from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.utils import (
def entanglement_of_formation(state: Statevector | DensityMatrix) -> float:
    """Calculate the entanglement of formation of quantum state.

    The input quantum state must be either a bipartite state vector, or a
    2-qubit density matrix.

    Args:
        state (Statevector or DensityMatrix): a 2-qubit quantum state.

    Returns:
        float: The entanglement of formation.

    Raises:
        QiskitError: if the input state is not a valid QuantumState.
        QiskitError: if input is not a bipartite QuantumState.
        QiskitError: if density matrix input is not a 2-qubit state.
    """
    state = _format_state(state, validate=True)
    if isinstance(state, Statevector):
        dims = state.dims()
        if len(dims) != 2:
            raise QiskitError('Input is not a bipartite quantum state.')
        qargs = [0] if dims[0] > dims[1] else [1]
        return entropy(partial_trace(state, qargs), base=2)
    if state.dim != 4:
        raise QiskitError('Input density matrix must be a 2-qubit state.')
    conc = concurrence(state)
    val = (1 + np.sqrt(1 - conc ** 2)) / 2
    return shannon_entropy([val, 1 - val])