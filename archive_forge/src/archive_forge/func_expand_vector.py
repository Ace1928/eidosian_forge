from collections.abc import Iterable
import functools
import inspect
import numbers
import numpy as np
import pennylane as qml
def expand_vector(vector, original_wires, expanded_wires):
    """Expand a vector to more wires.

    Args:
        vector (array): :math:`2^n` vector where n = len(original_wires).
        original_wires (Sequence[int]): original wires of vector
        expanded_wires (Union[Sequence[int], int]): expanded wires of vector, can be shuffled
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`2^m` vector where m = len(expanded_wires).
    """
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))
    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N
    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")
    if qml.math.shape(vector) != (2 ** N,):
        raise ValueError('Vector parameter must be of length 2**len(original_wires)')
    dims = [2] * N
    tensor = qml.math.reshape(vector, dims)
    if D > 0:
        extra_dims = [2] * D
        ones = qml.math.ones(2 ** D).reshape(extra_dims)
        expanded_tensor = qml.math.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor
    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))
    wire_indices = np.array(wire_indices)
    original_indices = np.array(range(N))
    expanded_tensor = qml.math.moveaxis(expanded_tensor, tuple(original_indices), tuple(wire_indices))
    return qml.math.reshape(expanded_tensor, 2 ** M)