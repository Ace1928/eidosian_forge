from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def lifted_state_operator(state: TensorProductState, qubits: List[int]) -> np.ndarray:
    """Take a TensorProductState along with a list of qubits and return a matrix
    corresponding to the tensored-up representation of the states' density operator form.

    Developer note: Quil and the QVM like qubits to be ordered such that qubit 0 is on the right.
    Therefore, in ``qubit_adjacent_lifted_gate``, ``lifted_pauli``, and ``lifted_state_operator``,
    we build up the lifted matrix by using the *left* kronecker product.

    :param state: The state
    :param qubits: list of qubits in the order they will be represented in the resultant matrix.
    """
    mat: np.ndarray = np.eye(1)
    for qubit in qubits:
        oneq_state = state[qubit]
        assert oneq_state.qubit == qubit
        state_vector = STATES[oneq_state.label][oneq_state.index][:, np.newaxis]
        state_matrix = state_vector @ state_vector.conj().T
        mat = np.kron(state_matrix, mat)
    return mat