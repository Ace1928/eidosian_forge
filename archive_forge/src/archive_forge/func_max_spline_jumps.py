from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def max_spline_jumps(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """Get maximum difference between spline and same branch polarization data."""
    tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
    sps = self.same_branch_splines(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
    max_jumps = [None, None, None]
    for i, sp in enumerate(sps):
        if sp is not None:
            max_jumps[i] = max(tot[:, i].ravel() - sp(range(len(tot[:, i].ravel()))))
    return max_jumps