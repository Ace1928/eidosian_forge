from typing import Sequence
import numpy as np
from cirq import protocols
def operation_to_choi(operation: 'protocols.SupportsKraus') -> np.ndarray:
    """Returns the unique Choi matrix associated with an operation .

    Choi matrix J(E) of a linear map E: L(H1) -> L(H2) which takes linear operators
    on Hilbert space H1 to linear operators on Hilbert space H2 is defined as

        $$
        J(E) = (E \\otimes I)(|\\phi\\rangle\\langle\\phi|)
        $$

    where $|\\phi\\rangle = \\sum_i|i\\rangle|i\\rangle$ is the unnormalized maximally
    entangled state and I: L(H1) -> L(H1) is the identity map. Note that J(E) is
    a square matrix with d1*d2 rows and columns where d1 = dim H1 and d2 = dim H2.

    Args:
        operation: Quantum channel.
    Returns:
        Choi matrix corresponding to operation.
    """
    return kraus_to_choi(protocols.kraus(operation))