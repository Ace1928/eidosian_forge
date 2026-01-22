from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def rectangle_crosses_area(self, d1, d2, a1, a2):
    """Whether a given rectangle crosses the area defined by the upper and lower curves.

        Args:
            d1: lower d.
            d2: upper d.
            a1: lower a.
            a2: upper a.
        """
    if d1 <= self.dmin and d2 <= self.dmin:
        return False
    if d1 >= self.dmax and d2 >= self.dmax:
        return False
    if d1 <= self.dmin and d2 <= self.dmax:
        ld2 = self.f_lower(d2)
        if a2 <= ld2 or a1 >= self.amax:
            return False
        return True
    if d1 <= self.dmin and d2 >= self.dmax:
        if a2 <= self.amin or a1 >= self.amax:
            return False
        return True
    if self.dmin <= d1 <= self.dmax and self.dmin <= d2 <= self.dmax:
        ld1 = self.f_lower(d1)
        ld2 = self.f_lower(d2)
        if a2 <= ld1 and a2 <= ld2:
            return False
        ud1 = self.f_upper(d1)
        ud2 = self.f_upper(d2)
        if a1 >= ud1 and a1 >= ud2:
            return False
        return True
    if self.dmin <= d1 <= self.dmax and d2 >= self.dmax:
        ud1 = self.f_upper(d1)
        if a1 >= ud1 or a2 <= self.amin:
            return False
        return True
    raise ValueError('Should not reach this point!')