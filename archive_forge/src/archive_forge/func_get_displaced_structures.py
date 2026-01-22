from __future__ import annotations
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import CompletePhononDos, PhononDos
from pymatgen.phonon.gruneisen import GruneisenParameter, GruneisenPhononBandStructureSymmLine
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from pymatgen.symmetry.bandstructure import HighSymmKpath
@requires(Phonopy, 'phonopy not installed!')
def get_displaced_structures(pmg_structure, atom_disp=0.01, supercell_matrix=None, yaml_fname=None, **kwargs):
    """
    Generate a set of symmetrically inequivalent displaced structures for
    phonon calculations.

    Args:
        pmg_structure (Structure): A pymatgen structure object.
        atom_disp (float): Atomic displacement. Default is 0.01 $\\\\AA$.
        supercell_matrix (3x3 array): Scaling matrix for supercell.
        yaml_fname (str): If not None, it represents the full path to
            the outputting displacement yaml file, e.g. disp.yaml.
        **kwargs: Parameters used in Phonopy.generate_displacement method.

    Returns:
        A list of symmetrically inequivalent structures with displacements, in
        which the first element is the perfect supercell structure.
    """
    is_plus_minus = kwargs.get('is_plusminus', 'auto')
    is_diagonal = kwargs.get('is_diagonal', True)
    is_trigonal = kwargs.get('is_trigonal', False)
    ph_structure = get_phonopy_structure(pmg_structure)
    if supercell_matrix is None:
        supercell_matrix = np.eye(3) * np.array((1, 1, 1))
    phonon = Phonopy(unitcell=ph_structure, supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=atom_disp, is_plusminus=is_plus_minus, is_diagonal=is_diagonal, is_trigonal=is_trigonal)
    if yaml_fname is not None:
        displacements = phonon.get_displacements()
        write_disp_yaml(displacements=displacements, supercell=phonon.get_supercell(), filename=yaml_fname)
    disp_supercells = phonon.get_supercells_with_displacements()
    init_supercell = phonon.get_supercell()
    structure_list = [get_pmg_structure(init_supercell)]
    for cell in disp_supercells:
        if cell is not None:
            structure_list.append(get_pmg_structure(cell))
    return structure_list