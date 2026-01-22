from __future__ import annotations
from typing import Union
import numpy as np
def bit_permutation_1q(n: int, k: int) -> np.ndarray:
    """
    Constructs index permutation that brings a circuit consisting of a single
    1-qubit gate to "standard form": ``kron(I(2^n/2), G)``, as we call it. Here n
    is the number of qubits, ``G`` is a 2x2 gate matrix, ``I(2^n/2)`` is the identity
    matrix of size ``(2^n/2)x(2^n/2)``, and the full size of the circuit matrix is
    ``(2^n)x(2^n)``. Circuit matrix in standard form becomes block-diagonal (with
    sub-matrices ``G`` on the main diagonal). Multiplication of such a matrix and
    a dense one is much faster than generic dense-dense product. Moreover,
    we do not need to keep the entire circuit matrix in memory but just 2x2 ``G``
    one. This saves a lot of memory when the number of qubits is large.

    Args:
        n: number of qubits.
        k: index of qubit where single 1-qubit gate is applied.

    Returns:
        permutation that brings the whole layer to the standard form.
    """
    perm = np.arange(2 ** n, dtype=np.int64)
    if k != n - 1:
        for v in range(2 ** n):
            perm[v] = swap_bits(v, k, n - 1)
    return perm