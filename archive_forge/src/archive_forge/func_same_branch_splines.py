from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def same_branch_splines(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """
        Fit splines to same branch polarization. This is used to assess any jumps
        in the same branch polarization.
        """
    tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
    L = tot.shape[0]
    try:
        sp_a = UnivariateSpline(range(L), tot[:, 0].ravel())
    except Exception:
        sp_a = None
    try:
        sp_b = UnivariateSpline(range(L), tot[:, 1].ravel())
    except Exception:
        sp_b = None
    try:
        sp_c = UnivariateSpline(range(L), tot[:, 2].ravel())
    except Exception:
        sp_c = None
    return (sp_a, sp_b, sp_c)