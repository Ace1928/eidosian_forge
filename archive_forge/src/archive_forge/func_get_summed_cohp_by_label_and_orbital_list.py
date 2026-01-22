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
def get_summed_cohp_by_label_and_orbital_list(self, label_list, orbital_list, divisor=1, summed_spin_channels=False):
    """Returns a COHP object that includes a summed COHP divided by divisor.

        Args:
            label_list: list of labels for the COHP that should be included in the summed cohp
            orbital_list: list of orbitals for the COHPs that should be included in the summed cohp (same order as
                label_list)
            divisor: float/int, the summed cohp will be divided by this divisor
            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            Returns a COHP object including a summed COHP
        """
    if not len(label_list) == len(orbital_list):
        raise ValueError("label_list and orbital_list don't have the same length!")
    first_cohpobject = self.get_orbital_resolved_cohp(label_list[0], orbital_list[0])
    summed_cohp = first_cohpobject.cohp.copy()
    summed_icohp = first_cohpobject.icohp.copy()
    for ilabel, label in enumerate(label_list[1:], start=1):
        cohp_here = self.get_orbital_resolved_cohp(label, orbital_list[ilabel])
        summed_cohp[Spin.up] = np.sum([summed_cohp[Spin.up], cohp_here.cohp.copy()[Spin.up]], axis=0)
        if Spin.down in summed_cohp:
            summed_cohp[Spin.down] = np.sum([summed_cohp[Spin.down], cohp_here.cohp.copy()[Spin.down]], axis=0)
        summed_icohp[Spin.up] = np.sum([summed_icohp[Spin.up], cohp_here.icohp.copy()[Spin.up]], axis=0)
        if Spin.down in summed_icohp:
            summed_icohp[Spin.down] = np.sum([summed_icohp[Spin.down], cohp_here.icohp.copy()[Spin.down]], axis=0)
    divided_cohp = {}
    divided_icohp = {}
    divided_cohp[Spin.up] = np.divide(summed_cohp[Spin.up], divisor)
    divided_icohp[Spin.up] = np.divide(summed_icohp[Spin.up], divisor)
    if Spin.down in summed_cohp:
        divided_cohp[Spin.down] = np.divide(summed_cohp[Spin.down], divisor)
        divided_icohp[Spin.down] = np.divide(summed_icohp[Spin.down], divisor)
    if summed_spin_channels and Spin.down in divided_cohp:
        final_cohp = {}
        final_icohp = {}
        final_cohp[Spin.up] = np.sum([divided_cohp[Spin.up], divided_cohp[Spin.down]], axis=0)
        final_icohp[Spin.up] = np.sum([divided_icohp[Spin.up], divided_icohp[Spin.down]], axis=0)
    else:
        final_cohp = divided_cohp
        final_icohp = divided_icohp
    return Cohp(efermi=first_cohpobject.efermi, energies=first_cohpobject.energies, cohp=final_cohp, are_coops=first_cohpobject.are_coops, are_cobis=first_cohpobject.are_cobis, icohp=final_icohp)