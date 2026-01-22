import numpy as np
from numpy import linalg
from ase import units 
def get_morse_potential_gradient(atoms, morse):
    i = morse.atomi
    j = morse.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    exp = np.exp(-morse.alpha * (dij - morse.r0))
    gr = 2.0 * morse.D * morse.alpha * exp * (1.0 - exp) * eij
    gx = np.dot(Mx.T, gr)
    morse.r = dij
    return (i, j, gx)