from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .dihedral import CNOTDihedral
def random_cnotdihedral(num_qubits, seed=None):
    """Return a random CNOTDihedral element.

    Args:
        num_qubits (int): the number of qubits for the CNOTDihedral object.
        seed (int or RandomState): Optional. Set a fixed seed or
                                   generator for RNG.
    Returns:
        CNOTDihedral: a random CNOTDihedral element.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    elem = CNOTDihedral(num_qubits=num_qubits)
    weight_1 = rng.integers(8, size=num_qubits)
    elem.poly.weight_1 = weight_1
    weight_2 = 2 * rng.integers(4, size=int(num_qubits * (num_qubits - 1) / 2))
    elem.poly.weight_2 = weight_2
    weight_3 = 4 * rng.integers(2, size=int(num_qubits * (num_qubits - 1) * (num_qubits - 2) / 6))
    elem.poly.weight_3 = weight_3
    from qiskit.synthesis.linear import random_invertible_binary_matrix
    linear = random_invertible_binary_matrix(num_qubits, seed=rng)
    elem.linear = linear
    shift = rng.integers(2, size=num_qubits)
    elem.shift = shift
    return elem