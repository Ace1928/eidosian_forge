import numpy as np
from numpy import linalg
from ase import units 
def get_angle_potential_value(atoms, angle):
    i = angle.atomi
    j = angle.atomj
    k = angle.atomk
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    rkj = rel_pos_pbc(atoms, k, j)
    dkj = linalg.norm(rkj)
    ekj = rkj / dkj
    eijekj = np.dot(eij, ekj)
    if np.abs(eijekj) > 1.0:
        eijekj = np.sign(eijekj)
    a = np.arccos(eijekj)
    if angle.cos:
        da = np.cos(a) - np.cos(angle.a0)
    else:
        da = a - angle.a0
        da = da - np.around(da / np.pi) * np.pi
    v = 0.5 * angle.k * da ** 2
    angle.a = a
    return (i, j, k, v)