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
def qwc_rotation(pauli_operators):
    """Performs circuit implementation of diagonalizing unitary for a Pauli word.

    Args:
        pauli_operators (list[Union[PauliX, PauliY, PauliZ, Identity]]): Single-qubit Pauli
            operations. No Pauli operations in this list may be acting on the same wire.
    Raises:
        TypeError: if any elements of ``pauli_operators`` are not instances of
            :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`, or :class:`~.Identity`

    **Example**

    >>> pauli_operators = [qml.X('a'), qml.Y('b'), qml.Z('c')]
    >>> qwc_rotation(pauli_operators)
    [RY(-1.5707963267948966, wires=['a']), RX(1.5707963267948966, wires=['b'])]
    """
    paulis_with_identity = (qml.Identity, qml.X, qml.Y, qml.Z)
    if not all((isinstance(element, paulis_with_identity) for element in pauli_operators)):
        raise TypeError(f'All values of input pauli_operators must be either Identity, PauliX, PauliY, or PauliZ instances, instead got pauli_operators = {pauli_operators}.')
    with OperationRecorder() as rec:
        for pauli in pauli_operators:
            if isinstance(pauli, qml.X):
                qml.RY(-np.pi / 2, wires=pauli.wires)
            elif isinstance(pauli, qml.Y):
                qml.RX(np.pi / 2, wires=pauli.wires)
    return rec.queue