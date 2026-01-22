import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def prep_symmetry(atoms, symprec=1e-06, verbose=False):
    """
    Prepare `at` for symmetry-preserving minimisation at precision `symprec`

    Returns a tuple `(rotations, translations, symm_map)`
    """
    import spglib
    dataset = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms), symprec=symprec)
    if verbose:
        print_symmetry(symprec, dataset)
    rotations = dataset['rotations'].copy()
    translations = dataset['translations'].copy()
    symm_map = []
    scaled_pos = atoms.get_scaled_positions()
    for rot, trans in zip(rotations, translations):
        this_op_map = [-1] * len(atoms)
        for i_at in range(len(atoms)):
            new_p = rot @ scaled_pos[i_at, :] + trans
            dp = scaled_pos - new_p
            dp -= np.round(dp)
            i_at_map = np.argmin(np.linalg.norm(dp, axis=1))
            this_op_map[i_at] = i_at_map
        symm_map.append(this_op_map)
    return (rotations, translations, symm_map)