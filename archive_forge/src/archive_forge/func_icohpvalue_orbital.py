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
def icohpvalue_orbital(self, orbitals, spin=Spin.up):
    """
        Args:
            orbitals: List of Orbitals or "str(Orbital1)-str(Orbital2)"
            spin: Spin.up or Spin.down.

        Returns:
            icohpvalue (float) corresponding to chosen spin.
        """
    if not self.is_spin_polarized and spin == Spin.down:
        raise ValueError('The calculation was not performed with spin polarization')
    if isinstance(orbitals, list):
        orbitals = f'{orbitals[0]}-{orbitals[1]}'
    return self._orbitals[orbitals]['icohp'][spin]