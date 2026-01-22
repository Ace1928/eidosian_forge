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
def extremum_icohpvalue(self, summed_spin_channels=True, spin=Spin.up):
    """Get ICOHP/ICOOP of strongest bond.

        Args:
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed.

            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned

        Returns:
            lowest ICOHP/largest ICOOP value (i.e. ICOHP/ICOOP value of strongest bond)
        """
    extremum = -sys.float_info.max if self._are_coops or self._are_cobis else sys.float_info.max
    if not self._is_spin_polarized:
        if spin == Spin.down:
            warnings.warn('This spin channel does not exist. I am switching to Spin.up')
        spin = Spin.up
    for value in self._icohplist.values():
        if not value.is_spin_polarized or not summed_spin_channels:
            if not self._are_coops and (not self._are_cobis):
                if value.icohpvalue(spin) < extremum:
                    extremum = value.icohpvalue(spin)
            elif value.icohpvalue(spin) > extremum:
                extremum = value.icohpvalue(spin)
        elif not self._are_coops and (not self._are_cobis):
            if value.summed_icohp < extremum:
                extremum = value.summed_icohp
        elif value.summed_icohp > extremum:
            extremum = value.summed_icohp
    return extremum