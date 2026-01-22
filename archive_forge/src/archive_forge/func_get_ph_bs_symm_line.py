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
def get_ph_bs_symm_line(bands_path, has_nac=False, labels_dict=None):
    """
    Creates a pymatgen PhononBandStructure from a band.yaml file.
    The labels will be extracted from the dictionary, if present.
    If the 'eigenvector'  key is found the eigendisplacements will be
    calculated according to the formula:
    \\\\exp(2*pi*i*(frac_coords \\\\dot q) / sqrt(mass) * v
     and added to the object.

    Args:
        bands_path: path to the band.yaml file
        has_nac: True if the data have been obtained with the option
            --nac option. Default False.
        labels_dict: dict that links a q-point in frac coords to a label.
    """
    return get_ph_bs_symm_line_from_dict(loadfn(bands_path), has_nac, labels_dict)