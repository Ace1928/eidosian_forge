import numpy as np
from numpy import linalg
from ase import units 
def get_dihedral_potential_reduced_hessian_test(atoms, dihedral):
    i, j, k, l, gx = get_dihedral_potential_gradient(atoms, dihedral)
    if dihedral.n is None:
        i, j, k, l, v = get_dihedral_potential_value(atoms, dihedral)
        Hx = np.tensordot(gx, gx, axes=0) / v / 2.0
    else:
        arg = dihedral.n * dihedral.d - dihedral.d0
        Hx = np.tensordot(gx, gx, axes=0) / dihedral.k / np.sin(arg) / np.sin(arg) * np.cos(arg)
    return (i, j, k, l, Hx)