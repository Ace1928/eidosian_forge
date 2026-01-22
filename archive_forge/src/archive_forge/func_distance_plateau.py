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
def distance_plateau(self):
    """Returns the distances plateau's for this NeighborsSet."""
    all_nbs_normalized_distances_sorted = sorted((nb['normalized_distance'] for nb in self.voronoi), reverse=True)
    maxdist = np.max(self.normalized_distances)
    plateau = None
    for idist, dist in enumerate(all_nbs_normalized_distances_sorted):
        if np.isclose(dist, maxdist, rtol=0.0, atol=self.detailed_voronoi.normalized_distance_tolerance):
            plateau = np.inf if idist == 0 else all_nbs_normalized_distances_sorted[idist - 1] - maxdist
            break
    if plateau is None:
        raise ValueError('Plateau not found ...')
    return plateau