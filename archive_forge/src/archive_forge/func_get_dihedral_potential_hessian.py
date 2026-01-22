import numpy as np
from numpy import linalg
from ase import units 
def get_dihedral_potential_hessian(atoms, dihedral, morses=None, spectral=False):
    eps = 1e-06
    i, j, k, l, g = get_dihedral_potential_gradient(atoms, dihedral)
    Hx = np.zeros((12, 12))
    dihedral_eps = Dihedral(dihedral.atomi, dihedral.atomj, dihedral.atomk, dihedral.atoml, dihedral.k, dihedral.d0, dihedral.n)
    indx = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2, 3 * k, 3 * k + 1, 3 * k + 2, 3 * l, 3 * l + 1, 3 * l + 2]
    for x in range(12):
        a = atoms.copy()
        positions = np.reshape(a.get_positions(), -1)
        positions[indx[x]] += eps
        a.set_positions(np.reshape(positions, (len(a), 3)))
        i, j, k, l, geps = get_dihedral_potential_gradient(a, dihedral_eps)
        for y in range(12):
            Hx[x, y] += 0.5 * (geps[y] - g[y]) / eps
            Hx[y, x] += 0.5 * (geps[y] - g[y]) / eps
    if dihedral.alpha is not None:
        rij = rel_pos_pbc(atoms, i, j)
        dij = linalg.norm(rij)
        rkj = rel_pos_pbc(atoms, k, j)
        dkj = linalg.norm(rkj)
        rkl = rel_pos_pbc(atoms, k, l)
        dkl = linalg.norm(rkl)
        Hx *= np.exp(dihedral.alpha[0] * (dihedral.rref[0] ** 2 - dij ** 2)) * np.exp(dihedral.alpha[1] * (dihedral.rref[1] ** 2 - dkj ** 2)) * np.exp(dihedral.alpha[2] * (dihedral.rref[2] ** 2 - dkl ** 2))
    if morses is not None:
        for m in range(len(morses)):
            if morses[m].atomi == i or morses[m].atomi == j or morses[m].atomi == k or (morses[m].atomi == l):
                Hx *= get_morse_potential_eta(atoms, morses[m])
            elif morses[m].atomj == i or morses[m].atomj == j or morses[m].atomj == k or (morses[m].atomj == l):
                Hx *= get_morse_potential_eta(atoms, morses[m])
    if spectral:
        eigvals, eigvecs = linalg.eigh(Hx)
        D = np.diag(np.abs(eigvals))
        U = eigvecs
        Hx = np.dot(U, np.dot(D, np.transpose(U)))
    return (i, j, k, l, Hx)