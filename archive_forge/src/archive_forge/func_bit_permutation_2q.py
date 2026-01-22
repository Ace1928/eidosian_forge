from __future__ import annotations
from typing import Union
import numpy as np
def bit_permutation_2q(n: int, j: int, k: int) -> np.ndarray:
    """
    Constructs index permutation that brings a circuit consisting of a single
    2-qubit gate to "standard form": ``kron(I(2^n/4), G)``, as we call it. Here ``n``
    is the number of qubits, ``G`` is a 4x4 gate matrix, ``I(2^n/4)`` is the identity
    matrix of size ``(2^n/4)x(2^n/4)``, and the full size of the circuit matrix is
    ``(2^n)x(2^n)``. Circuit matrix in standard form becomes block-diagonal (with
    sub-matrices ``G`` on the main diagonal). Multiplication of such a matrix and
    a dense one is much faster than generic dense-dense product. Moreover,
    we do not need to keep the entire circuit matrix in memory but just 4x4 ``G``
    one. This saves a lot of memory when the number of qubits is large.

    Args:
        n: number of qubits.
        j: index of control qubit where single 2-qubit gate is applied.
        k: index of target qubit where single 2-qubit gate is applied.

    Returns:
        permutation that brings the whole layer to the standard form.
    """
    dim = 2 ** n
    perm = np.arange(dim, dtype=np.int64)
    if j < n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, j, n - 2), k, n - 1)
        elif k == n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), j, n - 2)
        else:
            for v in range(dim):
                perm[v] = swap_bits(v, j, n - 2)
    elif j == n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(v, k, n - 1)
        else:
            pass
    elif k < n - 2:
        for v in range(dim):
            perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), k, n - 1)
    else:
        for v in range(dim):
            perm[v] = swap_bits(v, n - 2, n - 1)
    return perm