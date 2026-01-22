from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def program_unitary(program: Program, n_qubits: int) -> np.ndarray:
    """
    Return the unitary of a pyQuil program.

    :param program: A program consisting only of :py:class:`Gate`.:
    :return: a unitary corresponding to the composition of the program's gates.
    """
    umat: np.ndarray = np.eye(2 ** n_qubits)
    for instruction in program:
        if isinstance(instruction, Gate):
            unitary = lifted_gate(gate=instruction, n_qubits=n_qubits)
            umat = unitary.dot(umat)
        elif isinstance(instruction, Halt):
            pass
        else:
            raise ValueError(f'Can only compute program unitary for programs composed of `Gate`s. Found unsupported instruction: {instruction}')
    return umat