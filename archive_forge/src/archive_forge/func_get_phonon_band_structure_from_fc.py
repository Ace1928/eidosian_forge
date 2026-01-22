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
@requires(Phonopy, 'phonopy is required to calculate phonon band structures')
def get_phonon_band_structure_from_fc(structure: Structure, supercell_matrix: np.ndarray, force_constants: np.ndarray, mesh_density: float=100.0, **kwargs) -> PhononBandStructure:
    """
    Get a uniform phonon band structure from phonopy force constants.

    Args:
        structure: A structure.
        supercell_matrix: The supercell matrix used to generate the force
            constants.
        force_constants: The force constants in phonopy format.
        mesh_density: The density of the q-point mesh. See the docstring
            for the ``mesh`` argument in Phonopy.init_mesh() for more details.
        **kwargs: Additional kwargs passed to the Phonopy constructor.

    Returns:
        The uniform phonon band structure.
    """
    structure_phonopy = get_phonopy_structure(structure)
    phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix, **kwargs)
    phonon.set_force_constants(force_constants)
    phonon.run_mesh(mesh_density, is_mesh_symmetry=False, is_gamma_center=True)
    mesh = phonon.get_mesh_dict()
    return PhononBandStructure(mesh['qpoints'], mesh['frequencies'], structure.lattice)