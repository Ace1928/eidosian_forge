from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_polarization_change(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """Get difference between nonpolar and polar same branch polarization."""
    tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
    return (tot[-1] - tot[0]).reshape((1, 3))