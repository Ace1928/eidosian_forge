from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def is_qwc(pauli_vec_1, pauli_vec_2):
    """Checks if two Pauli words in the binary vector representation are qubit-wise commutative.

    Args:
        pauli_vec_1 (Union[list, tuple, array]): first binary vector argument in qubit-wise
            commutator
        pauli_vec_2 (Union[list, tuple, array]): second binary vector argument in qubit-wise
            commutator

    Returns:
        bool: returns True if the input Pauli words are qubit-wise commutative, returns False
        otherwise

    Raises:
        ValueError: if the input vectors are of different dimension, if the vectors are not of even
            dimension, or if the vector components are not strictly binary

    **Example**

    >>> is_qwc([1,0,0,1,1,0],[1,0,1,0,1,0])
    False
    >>> is_qwc([1,0,1,1,1,0],[1,0,0,1,1,0])
    True
    """
    if isinstance(pauli_vec_1, (list, tuple)):
        pauli_vec_1 = np.asarray(pauli_vec_1)
    if isinstance(pauli_vec_2, (list, tuple)):
        pauli_vec_2 = np.asarray(pauli_vec_2)
    if len(pauli_vec_1) != len(pauli_vec_2):
        raise ValueError(f'Vectors a and b must be the same dimension, instead got shapes {np.shape(pauli_vec_1)} and {np.shape(pauli_vec_2)}.')
    if len(pauli_vec_1) % 2 != 0:
        raise ValueError(f'Symplectic vector-space must have even dimension, instead got vectors of shape {np.shape(pauli_vec_1)}.')
    if not (np.array_equal(pauli_vec_1, pauli_vec_1.astype(bool)) and np.array_equal(pauli_vec_2, pauli_vec_2.astype(bool))):
        raise ValueError(f'Vectors a and b must have strictly binary components, instead got {pauli_vec_1} and {pauli_vec_2}')
    n_qubits = int(len(pauli_vec_1) / 2)
    for i in range(n_qubits):
        first_vec_ith_qubit_paulix = pauli_vec_1[i]
        first_vec_ith_qubit_pauliz = pauli_vec_1[n_qubits + i]
        second_vec_ith_qubit_paulix = pauli_vec_2[i]
        second_vec_ith_qubit_pauliz = pauli_vec_2[n_qubits + i]
        first_vec_qubit_i_is_identity = first_vec_ith_qubit_paulix == first_vec_ith_qubit_pauliz == 0
        second_vec_qubit_i_is_identity = second_vec_ith_qubit_paulix == second_vec_ith_qubit_pauliz == 0
        both_vecs_qubit_i_have_same_x = first_vec_ith_qubit_paulix == second_vec_ith_qubit_paulix
        both_vecs_qubit_i_have_same_z = first_vec_ith_qubit_pauliz == second_vec_ith_qubit_pauliz
        if not ((first_vec_qubit_i_is_identity or second_vec_qubit_i_is_identity) or (both_vecs_qubit_i_have_same_x and both_vecs_qubit_i_have_same_z)):
            return False
    return True