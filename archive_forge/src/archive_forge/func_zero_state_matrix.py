import warnings
from typing import Any, List, Optional, Sequence, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import P0, P1, KRAUS_OPS, QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate_matrix, lifted_gate, all_bitstrings
def zero_state_matrix(n_qubits: int) -> np.ndarray:
    """
    Construct a matrix corresponding to the tensor product of `n` ground states ``|0><0|``.

    :param n_qubits: The number of qubits.
    :return: The state matrix  ``|000...0><000...0|`` for `n_qubits`.
    """
    state_matrix = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=np.complex128)
    state_matrix[0, 0] = complex(1.0, 0)
    return state_matrix