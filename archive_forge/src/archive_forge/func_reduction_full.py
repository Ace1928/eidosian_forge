import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
def reduction_full(B):
    """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
    cycle_checker = CycleChecker(d=3)
    H = np.eye(3, dtype=int)
    norms = np.linalg.norm(B, axis=1)
    for it in range(MAX_IT):
        H = H[np.argsort(norms, kind='merge')]
        hw = H[2]
        hu, hv = reduction_gauss(B, H[0], H[1])
        H = np.array([hu, hv, hw])
        R = H @ B
        u, v, _ = R
        X = u / np.linalg.norm(u)
        Y = v - X * np.dot(v, X)
        Y /= np.linalg.norm(Y)
        pu, pv, pw = R @ np.array([X, Y]).T
        nb = closest_vector(pw, pu, pv)
        H[2] = [nb[0], nb[1], 1] @ H
        R = H @ B
        norms = np.linalg.norm(R, axis=1)
        if norms[2] >= norms[1] or cycle_checker.add_site(H):
            return (R, H)
    raise RuntimeError(f'Reduced basis not found after {MAX_IT} iterations')