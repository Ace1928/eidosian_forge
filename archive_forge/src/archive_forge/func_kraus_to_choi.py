from typing import Sequence
import numpy as np
from cirq import protocols
def kraus_to_choi(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the unique Choi matrix corresponding to a Kraus representation of a channel.

    Quantum channel E: L(H1) -> L(H2) may be described by a collection of operators A_i, called
    Kraus operators, such that

        $$
        E(\\rho) = \\sum_i A_i \\rho A_i^\\dagger.
        $$

    Kraus representation is not unique. Alternatively, E may be specified by its Choi matrix J(E)
    defined as

        $$
        J(E) = (E \\otimes I)(|\\phi\\rangle\\langle\\phi|)
        $$

    where $|\\phi\\rangle = \\sum_i|i\\rangle|i\\rangle$ is the unnormalized maximally entangled state
    and I: L(H1) -> L(H1) is the identity map. Choi matrix is unique for a given channel.

    The computation of the Choi matrix from a Kraus representation is essentially a reconstruction
    of a matrix from its eigendecomposition. It has the cost of O(kd**4) where k is the number of
    Kraus operators and d is the dimension of the input and output Hilbert space.

    Args:
        kraus_operators: Sequence of Kraus operators specifying a quantum channel.

    Returns:
        Choi matrix of the channel specified by kraus_operators.
    """
    d = np.prod(kraus_operators[0].shape, dtype=np.int64)
    choi_rank = len(kraus_operators)
    k = np.reshape(np.asarray(kraus_operators), (choi_rank, d))
    return np.einsum('bi,bj->ij', k, k.conj())