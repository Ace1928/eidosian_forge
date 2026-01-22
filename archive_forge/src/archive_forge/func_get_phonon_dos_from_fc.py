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
@requires(Phonopy, 'phonopy is required to calculate phonon density of states')
def get_phonon_dos_from_fc(structure: Structure, supercell_matrix: np.ndarray, force_constants: np.ndarray, mesh_density: float=100.0, num_dos_steps: int=200, **kwargs) -> CompletePhononDos:
    """
    Get a projected phonon density of states from phonopy force constants.

    Args:
        structure: A structure.
        supercell_matrix: The supercell matrix used to generate the force
            constants.
        force_constants: The force constants in phonopy format.
        mesh_density: The density of the q-point mesh. See the docstring
            for the ``mesh`` argument in Phonopy.init_mesh() for more details.
        num_dos_steps: Number of frequency steps in the energy grid.
        **kwargs: Additional kwargs passed to the Phonopy constructor.

    Returns:
        The density of states.
    """
    structure_phonopy = get_phonopy_structure(structure)
    phonon = Phonopy(structure_phonopy, supercell_matrix=supercell_matrix, **kwargs)
    phonon.set_force_constants(force_constants)
    phonon.run_mesh(mesh_density, is_mesh_symmetry=False, with_eigenvectors=True, is_gamma_center=True)
    frequencies = phonon.get_mesh_dict()['frequencies']
    freq_min = frequencies.min()
    freq_max = frequencies.max()
    freq_pitch = (freq_max - freq_min) / num_dos_steps
    phonon.run_projected_dos(freq_min=freq_min, freq_max=freq_max, freq_pitch=freq_pitch)
    dos_raw = phonon.projected_dos.get_partial_dos()
    p_doses = dict(zip(structure, dos_raw[1]))
    total_dos = PhononDos(dos_raw[0], dos_raw[1].sum(axis=0))
    return CompletePhononDos(structure, total_dos, p_doses)