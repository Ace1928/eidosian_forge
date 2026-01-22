from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
def clear_environments(self, conditions=None):
    """
        Get the clear environments in the structure.

        Args:
            conditions: Conditions to be checked for an environment to be "clear".

        Returns:
            list: Clear environments in this structure.
        """
    clear_envs_list = set()
    for isite in range(len(self.structure)):
        if self.coordination_environments[isite] is None:
            continue
        if len(self.coordination_environments[isite]) == 0:
            continue
        if self.site_has_clear_environment(isite=isite, conditions=conditions):
            ce = max(self.coordination_environments[isite], key=lambda x: x['ce_fraction'])
            clear_envs_list.add(ce['ce_symbol'])
    return list(clear_envs_list)