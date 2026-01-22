from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lmto import LMTOCopl
from pymatgen.io.lobster import Cohpcar
from pymatgen.util.coord import get_linear_interpolated_value
from pymatgen.util.due import Doi, due
from pymatgen.util.num import round_to_sigfigs
def get_cohp_by_label(self, label, summed_spin_channels=False):
    """Get specific COHP object.

        Args:
            label: string (for newer Lobster versions: a number)
            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            Returns the COHP object to simplify plotting
        """
    if label.lower() == 'average':
        divided_cohp = self.cohp
        divided_icohp = self.icohp
    else:
        divided_cohp = self.all_cohps[label].get_cohp(spin=None, integrated=False)
        divided_icohp = self.all_cohps[label].get_icohp(spin=None)
    if summed_spin_channels and Spin.down in self.cohp:
        final_cohp = {}
        final_icohp = {}
        final_cohp[Spin.up] = np.sum([divided_cohp[Spin.up], divided_cohp[Spin.down]], axis=0)
        final_icohp[Spin.up] = np.sum([divided_icohp[Spin.up], divided_icohp[Spin.down]], axis=0)
    else:
        final_cohp = divided_cohp
        final_icohp = divided_icohp
    return Cohp(efermi=self.efermi, energies=self.energies, cohp=final_cohp, are_coops=self.are_coops, are_cobis=self.are_cobis, icohp=final_icohp)