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
@property
def neighb_indices_and_images(self) -> list[dict[str, int]]:
    """List of indices and images with respect to the original unit cell sites for this NeighborsSet."""
    return [{'index': self.all_nbs_sites[inb]['index'], 'image_cell': self.all_nbs_sites[inb]['image_cell']} for inb in self.all_nbs_sites_indices_unsorted]