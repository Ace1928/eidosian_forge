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
def binary_to_pauli(binary_vector, wire_map=None):
    """Converts a binary vector of even dimension to an Observable instance.

    This functions follows the convention that the first half of binary vector components specify
    PauliX placements while the last half specify PauliZ placements.

    Args:
        binary_vector (Union[list, tuple, array]): binary vector of even dimension representing a
            unique Pauli word
        wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys, and
            unique integer labels as their values

    Returns:
        Union[Tensor, Prod]: The Pauli word corresponding to the input binary vector.
        Note that if a zero vector is input, then the resulting Pauli word will be
        an :class:`~.Identity` instance. If new operator arithmetic is enabled via
        :func:`~.pennylane.operation.enable_new_opmath`, a :class:`~.Prod` will be
        returned, else a :class:`~.Tensor` will be returned.

    Raises:
        TypeError: if length of binary vector is not even, or if vector does not have strictly
            binary components

    **Example**

    If ``wire_map`` is unspecified, the Pauli operations follow the same enumerations as the vector
    components, i.e., the ``i`` and ``N+i`` components specify the Pauli operation on wire ``i``,

    >>> binary_to_pauli([0,1,1,0,1,0])
    Tensor(Y(1), X(2))

    An arbitrary labelling can be assigned by using ``wire_map``:

    >>> wire_map = {'a': 0, 'b': 1, 'c': 2}
    >>> binary_to_pauli([0,1,1,0,1,0], wire_map=wire_map)
    Tensor(Y('b'), X('c'))

    Note that the values of ``wire_map``, if specified, must be ``0,1,..., N``,
    where ``N`` is the dimension of the vector divided by two, i.e.,
    ``list(wire_map.values())`` must be ``list(range(len(binary_vector)/2))``.
    """
    if isinstance(binary_vector, (list, tuple)):
        binary_vector = np.asarray(binary_vector)
    if len(binary_vector) % 2 != 0:
        raise ValueError(f'Length of binary_vector must be even, instead got vector of shape {np.shape(binary_vector)}.')
    if not np.array_equal(binary_vector, binary_vector.astype(bool)):
        raise ValueError(f'Input vector must have strictly binary components, instead got {binary_vector}.')
    n_qubits = len(binary_vector) // 2
    if wire_map is None:
        label_map = {i: i for i in range(n_qubits)}
    else:
        if set(wire_map.values()) != set(range(n_qubits)):
            raise ValueError(f'The values of wire_map must be integers 0 to N, for 2N-dimensional binary vector. Instead got wire_map values: {wire_map.values()}')
        label_map = {explicit_index: wire_label for wire_label, explicit_index in wire_map.items()}
    pauli_word = None
    for i in range(n_qubits):
        operation = None
        if binary_vector[i] == 1 and binary_vector[n_qubits + i] == 0:
            operation = PauliX(wires=Wires([label_map[i]]))
        elif binary_vector[i] == 1 and binary_vector[n_qubits + i] == 1:
            operation = PauliY(wires=Wires([label_map[i]]))
        elif binary_vector[i] == 0 and binary_vector[n_qubits + i] == 1:
            operation = PauliZ(wires=Wires([label_map[i]]))
        if operation is not None:
            if pauli_word is None:
                pauli_word = operation
            else:
                pauli_word @= operation
    if pauli_word is None:
        return Identity(wires=list(label_map.values())[0])
    return pauli_word