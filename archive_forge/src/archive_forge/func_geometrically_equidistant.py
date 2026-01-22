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
@classmethod
def geometrically_equidistant(cls, weight_cn1, weight_cn13):
    """Initialize geometrically equidistant weights for each coordination.

        Arge:
            weight_cn1: Weight of coordination 1.
            weight_cn13: Weight of coordination 13.

        Returns:
            CNBiasNbSetWeight.
        """
    initialization_options = {'type': 'geometrically_equidistant', 'weight_cn1': weight_cn1, 'weight_cn13': weight_cn13}
    factor = np.power(float(weight_cn13) / weight_cn1, 1 / 12.0)
    cn_weights = {cn: weight_cn1 * np.power(factor, cn - 1) for cn in range(1, 14)}
    return cls(cn_weights=cn_weights, initialization_options=initialization_options)