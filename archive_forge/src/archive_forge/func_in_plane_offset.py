from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@in_plane_offset.setter
def in_plane_offset(self, new_shift: np.ndarray) -> None:
    if len(new_shift) != 2:
        raise ValueError('In-plane shifts require two floats for a and b vectors')
    new_shift = np.mod(new_shift, 1)
    delta = new_shift - np.array(self.in_plane_offset)
    self._in_plane_offset = new_shift
    self.translate_sites(self.film_indices, [delta[0], delta[1], 0], to_unit_cell=True)