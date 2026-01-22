import numpy as np
from numpy import linalg
from ase import units 
def get_coulomb_potential_hessian(atoms, coulomb, spectral=False):
    i = coulomb.atomi
    j = coulomb.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Hr = 2.0 * coulomb.chargeij / dij ** 3 * Pij + -coulomb.chargeij / dij / dij / dij * Qij
    Hx = np.dot(Bx.T, np.dot(Hr, Bx))
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    coulomb.r = dij
    return (i, j, Hx)