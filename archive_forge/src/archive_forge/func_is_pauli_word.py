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
def is_pauli_word(observable):
    """
    Checks if an observable instance consists only of Pauli and Identity Operators.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Tensor` instance containing Pauli operators.

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a valid Pauli word.

    * A :class:`.Hamiltonian` instance with only one term.

    .. Warning::

        This function will only confirm that all operators are Pauli or Identity operators,
        and not whether the Observable is mathematically a Pauli word.
        If an Observable consists of multiple Pauli operators targeting the same wire, the
        function will return ``True`` regardless of any complex coefficients.


    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    **Example**

    >>> is_pauli_word(qml.Identity(0))
    True
    >>> is_pauli_word(qml.X(0) @ qml.Z(2))
    True
    >>> is_pauli_word(qml.Z(0) @ qml.Hadamard(1))
    False
    >>> is_pauli_word(4 * qml.X(0) @ qml.Z(0))
    True
    """
    return _is_pauli_word(observable) or len(observable.pauli_rep or []) == 1