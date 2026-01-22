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
def get_site_coordination_environments(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, return_maps=False):
    """Get the coordination environments of a given site.

        Args:
            site: Site for which coordination environment is needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.
            return_maps: Whether to return cn_maps (identifies all the NeighborsSet used).

        Returns:
            List of coordination environment.
        """
    if isite is None or dequivsite is None or dthissite is None or (mysym is None):
        isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
    return [self.get_site_coordination_environment(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_map=return_maps)]