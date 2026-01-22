import numpy as np
from numpy import linalg
from ase import units 
def get_dihedral_potential_reduced_hessian(atoms, dihedral, morses=None):
    i = dihedral.atomi
    j = dihedral.atomj
    k = dihedral.atomk
    l = dihedral.atoml
    rij = rel_pos_pbc(atoms, i, j)
    rkj = rel_pos_pbc(atoms, k, j)
    dkj = linalg.norm(rkj)
    dkj2 = dkj * dkj
    rkl = rel_pos_pbc(atoms, k, l)
    rijrkj = np.dot(rij, rkj)
    rkjrkl = np.dot(rkj, rkl)
    rmj = np.cross(rij, rkj)
    dmj = linalg.norm(rmj)
    dmj2 = dmj * dmj
    emj = rmj / dmj
    rnk = np.cross(rkj, rkl)
    dnk = linalg.norm(rnk)
    dnk2 = dnk * dnk
    enk = rnk / dnk
    emjenk = np.dot(emj, enk)
    if np.abs(emjenk) > 1.0:
        emjenk = np.sign(emjenk)
    d = np.sign(np.dot(rkj, np.cross(rmj, rnk))) * np.arccos(emjenk)
    dddri = dkj / dmj2 * rmj
    dddrl = -dkj / dnk2 * rnk
    gx = np.zeros(12)
    gx[0:3] = dddri
    gx[3:6] = (rijrkj / dkj2 - 1.0) * dddri - rkjrkl / dkj2 * dddrl
    gx[6:9] = (rkjrkl / dkj2 - 1.0) * dddrl - rijrkj / dkj2 * dddri
    gx[9:12] = dddrl
    if dihedral.d0 is None:
        Hx = np.abs(2.0 * dihedral.k * np.cos(2.0 * d)) * np.tensordot(gx, gx, axes=0)
    if dihedral.n is None:
        Hx = dihedral.k * np.tensordot(gx, gx, axes=0)
    else:
        Hx = np.abs(-dihedral.k * dihedral.n ** 2 * np.cos(dihedral.n * d - dihedral.d0)) * np.tensordot(gx, gx, axes=0)
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
    dihedral.d = d
    return (i, j, k, l, Hx)