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
def setup_random_structure(self, coordination):
    """
        Sets up a purely random structure with a given coordination.

        Args:
            coordination: coordination number for the random structure.
        """
    aa = 0.4
    bb = -0.2
    coords = []
    for _ in range(coordination + 1):
        coords.append(aa * np.random.random_sample(3) + bb)
    self.set_structure(lattice=np.array(np.eye(3) * 10, float), species=['Si'] * (coordination + 1), coords=coords, coords_are_cartesian=False)
    self.setup_random_indices_local_geometry(coordination)