from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def setup_local_geometry(self, isite, coords, optimization=None):
    """
        Sets up the AbstractGeometry for the local geometry of site with index isite.

        Args:
            isite: Index of the site for which the local geometry has to be set up
            coords: The coordinates of the (local) neighbors.
        """
    self.local_geometry = AbstractGeometry(central_site=self.structure.cart_coords[isite], bare_coords=coords, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid, optimization=optimization)