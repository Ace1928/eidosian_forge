from __future__ import annotations
from math import sqrt
import numpy as np
def two_qubit_local_invariants(U: np.ndarray) -> np.ndarray:
    """Computes the local invariants for a two-qubit unitary.

    Args:
        U (ndarray): Input two-qubit unitary.

    Returns:
        ndarray: NumPy array of local invariants [g0, g1, g2].

    Raises:
        ValueError: Input not a 2q unitary.

    Notes:
        Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
        Zhang et al., Phys Rev A. 67, 042313 (2003).
    """
    U = np.asarray(U)
    if U.shape != (4, 4):
        raise ValueError('Unitary must correspond to a two-qubit gate.')
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    det_um = np.linalg.det(Um)
    M = Um.T.dot(Um)
    m_tr2 = M.trace()
    m_tr2 *= m_tr2
    G1 = m_tr2 / (16 * det_um)
    G2 = (m_tr2 - np.trace(M.dot(M))) / (4 * det_um)
    return np.round([G1.real, G1.imag, G2.real], 12) + 0.0