import numpy as np
from numpy import linalg
from ase import units 
def get_morse_potential_hessian(atoms, morse, spectral=False):
    i = morse.atomi
    j = morse.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    exp = np.exp(-morse.alpha * (dij - morse.r0))
    Hr = 2.0 * morse.D * morse.alpha * exp * (morse.alpha * (2.0 * exp - 1.0) * Pij + (1.0 - exp) / dij * Qij)
    Hx = np.dot(Mx.T, np.dot(Hr, Mx))
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    morse.r = dij
    return (i, j, Hx)