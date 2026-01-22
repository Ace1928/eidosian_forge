import numpy as np
from numpy import linalg
from ase import units 
def get_bond_potential_hessian(atoms, bond, morses=None, spectral=False):
    i = bond.atomi
    j = bond.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    Qij = np.eye(3) - Pij
    Hr = bond.k * Pij + bond.k * (dij - bond.b0) / dij * Qij
    if bond.alpha is not None:
        Hr *= np.exp(bond.alpha[0] * (bond.rref[0] ** 2 - dij ** 2))
    if morses is not None:
        for m in range(len(morses)):
            if morses[m].atomi == i or morses[m].atomi == j:
                Hr *= get_morse_potential_eta(atoms, morses[m])
            elif morses[m].atomj == i or morses[m].atomj == j:
                Hr *= get_morse_potential_eta(atoms, morses[m])
    Hx = np.dot(Bx.T, np.dot(Hr, Bx))
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    bond.b = dij
    return (i, j, Hx)