import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def refine_symmetry(atoms, symprec=0.01, verbose=False):
    """
    Refine symmetry of an Atoms object

    Parameters
    ----------
    atoms - input Atoms object
    symprec - symmetry precicion
    verbose - if True, print out symmetry information before and after

    Returns
    -------

    spglib dataset

    """
    import spglib
    dataset = check_symmetry(atoms, symprec, verbose=verbose)
    std_cell = dataset['std_lattice']
    trans_std_cell = dataset['transformation_matrix'].T @ std_cell
    rot_trans_std_cell = trans_std_cell @ dataset['std_rotation_matrix']
    atoms.set_cell(rot_trans_std_cell, True)
    dataset = check_symmetry(atoms, symprec=symprec, verbose=verbose)
    res = spglib.find_primitive(atoms_to_spglib_cell(atoms), symprec=symprec)
    prim_cell, prim_scaled_pos, prim_types = res
    std_cell = dataset['std_lattice']
    rot_std_cell = std_cell @ dataset['std_rotation_matrix']
    rot_std_pos = dataset['std_positions'] @ rot_std_cell
    pos = atoms.get_positions()
    dp0 = pos[list(dataset['mapping_to_primitive']).index(0)] - rot_std_pos[list(dataset['std_mapping_to_primitive']).index(0)]
    rot_prim_cell = prim_cell @ dataset['std_rotation_matrix']
    inv_rot_prim_cell = np.linalg.inv(rot_prim_cell)
    aligned_std_pos = rot_std_pos + dp0
    mapping_to_primitive = list(dataset['mapping_to_primitive'])
    std_mapping_to_primitive = list(dataset['std_mapping_to_primitive'])
    pos = atoms.get_positions()
    for i_at in range(len(atoms)):
        std_i_at = std_mapping_to_primitive.index(mapping_to_primitive[i_at])
        dp = aligned_std_pos[std_i_at] - pos[i_at]
        dp_s = dp @ inv_rot_prim_cell
        pos[i_at] = aligned_std_pos[std_i_at] - np.round(dp_s) @ rot_prim_cell
    atoms.set_positions(pos)
    return check_symmetry(atoms, symprec=0.0001, verbose=verbose)