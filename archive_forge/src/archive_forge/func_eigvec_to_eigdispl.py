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
def eigvec_to_eigdispl(eig_vec, q, frac_coords, mass):
    """
    Converts a single eigenvector to an eigendisplacement in the primitive cell
    according to the formula:

        exp(2*pi*i*(frac_coords dot q) / sqrt(mass) * v

    Compared to the modulation option in phonopy, here all the additional
    multiplicative and phase factors are set to 1.

    Args:
        v: the vector that should be converted. A 3D complex numpy array.
        q: the q point in fractional coordinates
        frac_coords: the fractional coordinates of the atom
        mass: the mass of the atom
    """
    c = np.exp(2j * np.pi * np.dot(frac_coords, q)) / np.sqrt(mass)
    return c * eig_vec