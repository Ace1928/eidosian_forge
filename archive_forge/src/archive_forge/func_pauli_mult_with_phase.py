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
def pauli_mult_with_phase(pauli_1, pauli_2, wire_map=None):
    """Multiply two Pauli words together, and return both their product as a Pauli word
    and the global phase.

    .. warning::

        ``pauli_mult_with_phase`` is deprecated. Instead, you can multiply two Pauli
        words together with ``qml.simplify(qml.prod(pauli_1, pauli_2))``. Note that if
        there is a phase, this will be in ``result.scalar``, and the base will be
        available in ``result.base``.

    Two Pauli operations can be multiplied together by taking the additive
    OR of their binary symplectic representations. The phase is computed by
    looking at the number of times we have the products  :math:`XY, YZ`, or :math:`ZX` (adds a
    phase of :math:`i`), or :math:`YX, ZY, XZ` (adds a phase of :math:`-i`).

    Args:
        pauli_1 (.Operation): A Pauli word.
        pauli_2 (.Operation): A Pauli word to multiply with the first one.
        wire_map  (dict[Union[str, int], int]): dictionary containing all wire labels used in the Pauli
            word as keys, and unique integer labels as their values. If no wire map is
            provided, the map will be constructed from the set of wires acted on
            by the input Pauli words.

    Returns:
        tuple[.Operation, complex]: The product of ``pauli_1`` and ``pauli_2``, and the
        global phase.

    **Example**

    This function works the same as :func:`~.pauli_mult` but also returns the global
    phase accumulated as a result of the ordering of Paulis in the product (e.g., :math:`XY = iZ`,
    and :math:`YX = -iZ`).

    >>> from pennylane.pauli import pauli_mult_with_phase
    >>> pauli_1 = qml.X(0) @ qml.Z(1)
    >>> pauli_2 = qml.Y(0) @ qml.Z(1)
    >>> product, phase = pauli_mult_with_phase(pauli_1, pauli_2)
    >>> product
    Z(0)
    >>> phase
    1j
    """
    warn('`pauli_mult_with_phase` is deprecated. Instead, you can multiply two Pauli words together with `qml.simplify(qml.prod(pauli_1, pauli_2))`. Note that if there is a phase, this will be in `result.scalar`, and the base will be available in `result.base`.', qml.PennyLaneDeprecationWarning)
    if wire_map is None:
        wire_map = _wire_map_from_pauli_pair(pauli_1, pauli_2)
    pauli_product = pauli_mult(pauli_1, pauli_2, wire_map)
    pauli_1_string = pauli_word_to_string(pauli_1, wire_map=wire_map)
    pauli_2_string = pauli_word_to_string(pauli_2, wire_map=wire_map)
    pos_phases = (('X', 'Y'), ('Y', 'Z'), ('Z', 'X'))
    phase = 1
    for qubit_1_char, qubit_2_char in zip(pauli_1_string, pauli_2_string):
        if qubit_1_char == 'I' or qubit_2_char == 'I':
            continue
        if qubit_1_char == qubit_2_char:
            continue
        if (qubit_1_char, qubit_2_char) in pos_phases:
            phase *= 1j
        else:
            phase *= -1j
    return (pauli_product, phase)