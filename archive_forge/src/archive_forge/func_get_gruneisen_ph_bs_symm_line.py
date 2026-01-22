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
def get_gruneisen_ph_bs_symm_line(gruneisen_path, structure=None, structure_path=None, labels_dict=None, fit=False):
    """
    Creates a pymatgen GruneisenPhononBandStructure from a band.yaml file.
    The labels will be extracted from the dictionary, if present.
    If the 'eigenvector' key is found the eigendisplacements will be
    calculated according to the formula:
    \\\\exp(2*pi*i*(frac_coords \\\\dot q) / sqrt(mass) * v
     and added to the object.

    Args:
        gruneisen_path: path to the band.yaml file
        structure: pymaten Structure object
        structure_path: path to a structure file (e.g., POSCAR)
        labels_dict: dict that links a qpoint in frac coords to a label.
        fit: Substitute Grueneisen parameters close to the gamma point
            with points obtained from a fit to a spline if the derivate from
            a smooth curve (i.e. if the slope changes by more than 200% in the
            range of 10% around the gamma point).
            These derivations occur because of very small frequencies
            (and therefore numerical inaccuracies) close to gamma.
    """
    return get_gs_ph_bs_symm_line_from_dict(loadfn(gruneisen_path), structure, structure_path, labels_dict, fit)