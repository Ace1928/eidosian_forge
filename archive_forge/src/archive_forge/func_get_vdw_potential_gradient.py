import numpy as np
from numpy import linalg
from ase import units 
def get_vdw_potential_gradient(atoms, vdw):
    i = vdw.atomi
    j = vdw.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    gr = (-12.0 * vdw.Aij / dij ** 13 + 6.0 * vdw.Bij / dij ** 7) * eij
    gx = np.dot(Bx.T, gr)
    vdw.r = dij
    return (i, j, gx)