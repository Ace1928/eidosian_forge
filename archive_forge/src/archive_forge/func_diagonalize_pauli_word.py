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
def diagonalize_pauli_word(pauli_word):
    """Transforms the Pauli word to diagonal form in the computational basis.

    Args:
        pauli_word (Observable): the Pauli word to diagonalize in computational basis

    Returns:
        Observable: the Pauli word diagonalized in the computational basis

    Raises:
        TypeError: if the input is not a Pauli word, i.e., a Pauli operator,
            :class:`~.Identity`, or :class:`~.Tensor` instances thereof

    **Example**

    >>> diagonalize_pauli_word(qml.X('a') @ qml.Y('b') @ qml.Z('c'))
    Z('a') @ Z('b') @ Z('c')
    """
    if not is_pauli_word(pauli_word):
        raise TypeError(f'Input must be a Pauli word, instead got: {pauli_word}.')
    paulis_with_identity = (qml.X, qml.Y, qml.Z, qml.Identity)
    diag_term = None
    if isinstance(pauli_word, Tensor):
        for sigma in pauli_word.obs:
            if sigma.name != 'Identity':
                if diag_term is None:
                    diag_term = qml.Z(sigma.wires)
                else:
                    diag_term @= qml.Z(sigma.wires)
    elif isinstance(pauli_word, paulis_with_identity):
        sigma = pauli_word
        if sigma.name != 'Identity':
            diag_term = qml.Z(sigma.wires)
    if diag_term is None:
        diag_term = qml.Identity(pauli_word.wires.tolist()[0])
    return diag_term