from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
def random_clifford(num_qubits: int, seed: int | np.random.Generator | None=None):
    """Return a random Clifford operator.

    The Clifford is sampled using the method of Reference [1].

    Args:
        num_qubits (int): the number of qubits for the Clifford
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Clifford: a random Clifford operator.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    had, perm = _sample_qmallows(num_qubits, rng)
    gamma1 = np.diag(rng.integers(2, size=num_qubits, dtype=np.int8))
    gamma2 = np.diag(rng.integers(2, size=num_qubits, dtype=np.int8))
    delta1 = np.eye(num_qubits, dtype=np.int8)
    delta2 = delta1.copy()
    _fill_tril(gamma1, rng, symmetric=True)
    _fill_tril(gamma2, rng, symmetric=True)
    _fill_tril(delta1, rng)
    _fill_tril(delta2, rng)
    block_inverse_threshold = 50
    zero = np.zeros((num_qubits, num_qubits), dtype=np.int8)
    prod1 = np.matmul(gamma1, delta1) % 2
    prod2 = np.matmul(gamma2, delta2) % 2
    inv1 = _inverse_tril(delta1, block_inverse_threshold).transpose()
    inv2 = _inverse_tril(delta2, block_inverse_threshold).transpose()
    table1 = np.block([[delta1, zero], [prod1, inv1]])
    table2 = np.block([[delta2, zero], [prod2, inv2]])
    table = table2[np.concatenate([perm, num_qubits + perm])]
    inds = had * np.arange(1, num_qubits + 1)
    inds = inds[inds > 0] - 1
    lhs_inds = np.concatenate([inds, inds + num_qubits])
    rhs_inds = np.concatenate([inds + num_qubits, inds])
    table[lhs_inds, :] = table[rhs_inds, :]
    tableau = np.zeros((2 * num_qubits, 2 * num_qubits + 1), dtype=bool)
    tableau[:, :-1] = np.mod(np.matmul(table1, table), 2)
    tableau[:, -1] = rng.integers(2, size=2 * num_qubits)
    return Clifford(tableau, validate=False)