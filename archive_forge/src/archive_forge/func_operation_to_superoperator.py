from typing import Sequence
import numpy as np
from cirq import protocols
def operation_to_superoperator(operation: 'protocols.SupportsKraus') -> np.ndarray:
    """Returns the matrix representation of an operation in standard basis.

    Let E: L(H1) -> L(H2) denote a linear map which takes linear operators on Hilbert space H1
    to linear operators on Hilbert space H2 and let d1 = dim H1 and d2 = dim H2. Also, let Fij
    denote an operator whose matrix has one in ith row and jth column and zeros everywhere else.
    Note that d1-by-d1 operators Fij form a basis of L(H1). Similarly, d2-by-d2 operators Fij
    form a basis of L(H2). This function returns the matrix of E in these bases.

    Args:
        operation: Quantum channel.
    Returns:
        Matrix representation of operation.
    """
    return kraus_to_superoperator(protocols.kraus(operation))