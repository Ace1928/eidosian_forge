import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def symmetrize_rank1(lattice, inv_lattice, forces, rot, trans, symm_map):
    """
    Return symmetrized forces

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (reciprocal().T)
    """
    scaled_symmetrized_forces_T = np.zeros(forces.T.shape)
    scaled_forces_T = np.dot(inv_lattice.T, forces.T)
    for r, t, this_op_map in zip(rot, trans, symm_map):
        transformed_forces_T = np.dot(r, scaled_forces_T)
        scaled_symmetrized_forces_T[:, this_op_map] += transformed_forces_T
    scaled_symmetrized_forces_T /= len(rot)
    symmetrized_forces = (lattice.T @ scaled_symmetrized_forces_T).T
    return symmetrized_forces