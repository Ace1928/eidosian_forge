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
def pauli_mult(pauli_1, pauli_2, wire_map=None):
    """Multiply two Pauli words together and return the product as a Pauli word.

    .. warning::

        ``pauli_mult`` is deprecated. Instead, you can multiply two Pauli words
        together with ``qml.simplify(qml.prod(pauli_1, pauli_2))``. Note that if
        there is a phase, this will be in ``result.scalar``, and the base will be
        available in ``result.base``.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations.

    Args:
        pauli_1 (.Operation): A Pauli word.
        pauli_2 (.Operation): A Pauli word to multiply with the first one.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in the Pauli
            word as keys, and unique integer labels as their values. If no wire map is
            provided, the map will be constructed from the set of wires acted on
            by the input Pauli words.

    Returns:
        .Operation: The product of pauli_1 and pauli_2 as a Pauli word
        (ignoring the global phase).

    **Example**

    This function enables multiplication of Pauli group elements at the level of
    Pauli words, rather than matrices. For example,

    >>> from pennylane.pauli import pauli_mult
    >>> pauli_1 = qml.X(0) @ qml.Z(1)
    >>> pauli_2 = qml.Y(0) @ qml.Z(1)
    >>> product = pauli_mult(pauli_1, pauli_2)
    >>> print(product)
    Z(0)
    """
    warn('`pauli_mult` is deprecated. Instead, you can multiply two Pauli words together with `qml.simplify(qml.prod(pauli_1, pauli_2))`. Note that if there is a phase, this will be in `result.scalar`, and the base will be available in `result.base`.', qml.PennyLaneDeprecationWarning)
    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_1, pauli_2)
    if are_identical_pauli_words(pauli_1, pauli_2):
        first_wire = list(pauli_1.wires)[0]
        return Identity(first_wire)
    pauli_1_binary = pauli_to_binary(pauli_1, wire_map=wire_map)
    pauli_2_binary = pauli_to_binary(pauli_2, wire_map=wire_map)
    bin_symp_1 = np.array([int(x) for x in pauli_1_binary])
    bin_symp_2 = np.array([int(x) for x in pauli_2_binary])
    pauli_product = bin_symp_1 ^ bin_symp_2
    return binary_to_pauli(pauli_product, wire_map=wire_map)