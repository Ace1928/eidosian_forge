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
def pauli_word_to_matrix(pauli_word, wire_map=None):
    """Convert a Pauli word from a tensor to its matrix representation.

    The matrix representation of a Pauli word has dimension :math:`2^n \\times 2^n`,
    where :math:`n` is the number of qubits provided in ``wire_map``. For wires
    that the Pauli word does not act on, identities must be inserted into the tensor
    product at the correct positions.

    Args:
        pauli_word (Observable): an observable, either a :class:`~.Tensor`, :class:`~.Prod` or
            single-qubit observable representing a Pauli group element.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels used in
            the Pauli word as keys, and unique integer labels as their values

    Returns:
        array[complex]: The matrix representation of the multi-qubit Pauli over the
        specified wire map.

    Raises:
        TypeError: if the input observable is not a proper Pauli word.

    **Example**

    >>> wire_map = {'a' : 0, 'b' : 1}
    >>> pauli_word = qml.X('a') @ qml.Y('b')
    >>> pauli_word_to_matrix(pauli_word, wire_map=wire_map)
    array([[0.+0.j, 0.-0.j, 0.+0.j, 0.-1.j],
           [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
           [0.+0.j, 0.-1.j, 0.+0.j, 0.-0.j],
           [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    if not is_pauli_word(pauli_word):
        raise TypeError(f'Expected Pauli word observables, instead got {pauli_word}')
    if wire_map is None:
        wire_map = {pauli_word.wires.labels[i]: i for i in range(len(pauli_word.wires))}
    n_qubits = len(wire_map)
    if n_qubits == 1:
        return pauli_word.matrix()
    pauli_names = [pauli_word.name] if isinstance(pauli_word.name, str) else pauli_word.name
    if pauli_names == ['Identity']:
        return np.eye(2 ** n_qubits)
    pauli_mats = [ID_MAT for x in range(n_qubits)]
    for wire_label, wire_idx in wire_map.items():
        if wire_label in pauli_word.wires.labels:
            op_idx = pauli_word.wires.labels.index(wire_label)
            pauli_mats[wire_idx] = getattr(qml, pauli_names[op_idx]).compute_matrix()
    return reduce(np.kron, pauli_mats)