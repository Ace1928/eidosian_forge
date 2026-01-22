from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def calc_ionic(site: PeriodicSite, structure: Structure, zval: float) -> np.ndarray:
    """
    Calculate the ionic dipole moment using ZVAL from pseudopotential.

    site: PeriodicSite
    structure: Structure
    zval: Charge value for ion (ZVAL for VASP pseudopotential)

    Returns polarization in electron Angstroms.
    """
    norms = structure.lattice.lengths
    return np.multiply(norms, -site.frac_coords * zval)