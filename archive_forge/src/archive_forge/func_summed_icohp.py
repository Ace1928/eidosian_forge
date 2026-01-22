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
@property
def summed_icohp(self):
    """Sums ICOHPs of both spin channels for spin polarized compounds.

        Returns:
            float: icohp value in eV.
        """
    return self._icohp[Spin.down] + self._icohp[Spin.up] if self._is_spin_polarized else self._icohp[Spin.up]