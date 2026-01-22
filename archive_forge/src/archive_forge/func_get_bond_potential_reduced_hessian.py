import numpy as np
from numpy import linalg
from ase import units 
def get_bond_potential_reduced_hessian(atoms, bond, morses=None):
    i = bond.atomi
    j = bond.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    Hr = bond.k * Pij
    if bond.alpha is not None:
        Hr *= np.exp(bond.alpha[0] * (bond.rref[0] ** 2 - dij ** 2))
    if morses is not None:
        for m in range(len(morses)):
            if morses[m].atomi == i or morses[m].atomi == j:
                Hr *= get_morse_potential_eta(atoms, morses[m])
            elif morses[m].atomj == i or morses[m].atomj == j:
                Hr *= get_morse_potential_eta(atoms, morses[m])
    Hx = np.dot(Bx.T, np.dot(Hr, Bx))
    bond.b = dij
    return (i, j, Hx)