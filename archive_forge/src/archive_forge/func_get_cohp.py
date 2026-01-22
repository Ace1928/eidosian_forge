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
def get_cohp(self, spin=None, integrated=False):
    """Returns the COHP or ICOHP for a particular spin.

        Args:
            spin: Spin. Can be parsed as spin object, integer (-1/1)
                or str ("up"/"down")
            integrated: Return COHP (False) or ICOHP (True)

        Returns:
            Returns the CHOP or ICOHP for the input spin. If Spin is
            None and both spins are present, both spins will be returned
            as a dictionary.
        """
    populations = self.cohp if not integrated else self.icohp
    if populations is None:
        return None
    if spin is None:
        return populations
    if isinstance(spin, int):
        spin = Spin(spin)
    elif isinstance(spin, str):
        spin = Spin({'up': 1, 'down': -1}[spin.lower()])
    return {spin: populations[spin]}