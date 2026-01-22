import numpy as np
from numpy import linalg
from ase import units 
def get_angle_potential_reduced_hessian_test(atoms, angle):
    i, j, k, v = get_angle_potential_value(atoms, angle)
    i, j, k, gx = get_angle_potential_gradient(atoms, angle)
    Hx = np.tensordot(gx, gx, axes=0) / v / 2.0
    return (i, j, k, Hx)