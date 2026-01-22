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
def get_icohp_dict_by_bondlengths(self, minbondlength=0.0, maxbondlength=8.0):
    """Get a dict of IcohpValues corresponding to certain bond lengths.

        Args:
            minbondlength: defines the minimum of the bond lengths of the bonds
            maxbondlength: defines the maximum of the bond lengths of the bonds.

        Returns:
            dict of IcohpValues, the keys correspond to the values from the initial list_labels.
        """
    new_icohp_dict = {}
    for value in self._icohplist.values():
        if value._length >= minbondlength and value._length <= maxbondlength:
            new_icohp_dict[value._label] = value
    return new_icohp_dict