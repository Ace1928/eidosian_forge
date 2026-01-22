import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def pauli_expval(bits, recipes, word):
    """
    The approximate expectation value of a Pauli word given the bits and recipes
    from a classical shadow measurement.

    The expectation value can be computed using

    .. math::

        \\alpha = \\frac{1}{|T_{match}|}\\sum_{T_{match}}\\left(1 - 2\\left(\\sum b \\text{  mod }2\\right)\\right)

    where :math:`T_{match}` denotes the snapshots with recipes that match the Pauli word,
    and the right-most sum is taken over all bits in the snapshot where the observable
    in the Pauli word for that bit is not the identity.

    Args:
        bits (tensor-like[int]): An array with shape ``(T, n)``, where ``T`` is the
            number of snapshots and ``n`` is the number of measured qubits. Each
            entry must be either ``0`` or ``1`` depending on the sample for the
            corresponding snapshot and qubit.
        recipes (tensor-like[int]): An array with shape ``(T, n)``. Each entry
            must be either ``0``, ``1``, or ``2`` depending on the selected Pauli
            measurement for the corresponding snapshot and qubit. ``0`` corresponds
            to PauliX, ``1`` to PauliY, and ``2`` to PauliZ.
        word (tensor-like[int]): An array with shape ``(n,)``. Each entry must be
            either ``0``, ``1``, ``2``, or ``-1`` depending on the Pauli observable
            on each qubit. For example, when ``n=3``, the observable ``Y(0) @ X(2)``
            corresponds to the word ``np.array([1 -1 0])``.

    Returns:
        tensor-like[float]: An array with shape ``(T,)`` containing the value
        of the Pauli observable for each snapshot. The expectation can be
        found by averaging across the snapshots.
    """
    T, n = recipes.shape
    b = word.shape[0]
    bits = qml.math.cast(bits, np.int64)
    recipes = qml.math.cast(recipes, np.int64)
    word = qml.math.convert_like(qml.math.cast_like(word, bits), bits)
    id_mask = word == -1
    indices = qml.math.equal(qml.math.reshape(recipes, (T, 1, n)), qml.math.reshape(word, (1, b, n)))
    indices = np.logical_or(indices, qml.math.tile(qml.math.reshape(id_mask, (1, b, n)), (T, 1, 1)))
    indices = qml.math.all(indices, axis=2)
    bits = qml.math.where(id_mask, 0, qml.math.tile(qml.math.expand_dims(bits, 1), (1, b, 1)))
    bits = qml.math.sum(bits, axis=2) % 2
    expvals = qml.math.where(indices, 1 - 2 * bits, 0) * 3 ** np.count_nonzero(np.logical_not(id_mask), axis=1)
    return qml.math.cast(expvals, np.float64)