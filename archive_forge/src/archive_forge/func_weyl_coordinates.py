from __future__ import annotations
import numpy as np
def weyl_coordinates(U: np.ndarray) -> np.ndarray:
    """Computes the Weyl coordinates for a given two-qubit unitary matrix.

    Args:
        U (np.ndarray): Input two-qubit unitary.

    Returns:
        np.ndarray: Array of the 3 Weyl coordinates.
    """
    import scipy.linalg as la
    pi2 = np.pi / 2
    pi4 = np.pi / 4
    U = U / la.det(U) ** 0.25
    Up = transform_to_magic_basis(U, reverse=True)
    D = la.eigvals(Up.T @ Up)
    d = -np.angle(D) / 2
    d[3] = -d[0] - d[1] - d[2]
    cs = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)
    cstemp = np.mod(cs, pi2)
    np.minimum(cstemp, pi2 - cstemp, cstemp)
    order = np.argsort(cstemp)[[1, 2, 0]]
    cs = cs[order]
    d[:3] = d[order]
    if cs[0] > pi2:
        cs[0] -= 3 * pi2
    if cs[1] > pi2:
        cs[1] -= 3 * pi2
    conjs = 0
    if cs[0] > pi4:
        cs[0] = pi2 - cs[0]
        conjs += 1
    if cs[1] > pi4:
        cs[1] = pi2 - cs[1]
        conjs += 1
    if cs[2] > pi2:
        cs[2] -= 3 * pi2
    if conjs == 1:
        cs[2] = pi2 - cs[2]
    if cs[2] > pi4:
        cs[2] -= pi2
    return cs[[1, 0, 2]]