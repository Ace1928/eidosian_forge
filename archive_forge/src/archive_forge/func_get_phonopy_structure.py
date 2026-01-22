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
def get_phonopy_structure(pmg_structure: Structure) -> PhonopyAtoms:
    """
    Convert a pymatgen Structure object to a PhonopyAtoms object.

    Args:
        pmg_structure (pymatgen Structure): A Pymatgen structure object.
    """
    symbols = [site.specie.symbol for site in pmg_structure]
    return PhonopyAtoms(symbols=symbols, cell=pmg_structure.lattice.matrix, scaled_positions=pmg_structure.frac_coords, magnetic_moments=pmg_structure.site_properties.get('magmom'))