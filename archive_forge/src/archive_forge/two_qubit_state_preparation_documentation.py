from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops, qis, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
Prepares the given 2q state from |00> using at-most 1 ISWAP gate + single qubit rotations.

    Entangled states are prepared using exactly 1 ISWAP gate while product states are prepared
    using only single qubit rotations (0 ISWAP gates)

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        state: 4x1 matrix representing two qubit state vector, ordered as 00, 01, 10, 11.
        use_iswap_inv: If True, uses `cirq.ISWAP_INV` instead of `cirq.ISWAP`.

    Returns:
        List of operations (at-most 1 ISWAP + single qubit rotations) preparing state from |00>.
    