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
def get_ph_dos(total_dos_path):
    """
    Creates a pymatgen PhononDos from a total_dos.dat file.

    Args:
        total_dos_path: path to the total_dos.dat file.
    """
    a = np.loadtxt(total_dos_path)
    return PhononDos(a[:, 0], a[:, 1])