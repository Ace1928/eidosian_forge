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
def coordination_geometry_symmetry_measures_fallback_random(self, coordination_geometry, NRANDOM=10, points_perfect=None):
    """
        Returns the symmetry measures for a random set of permutations for the coordination geometry
        "coordination_geometry". Fallback implementation for the plane separation algorithms measures
        of each permutation

        Args:
            coordination_geometry: The coordination geometry to be investigated
            NRANDOM: Number of random permutations to be tested

        Returns:
            The symmetry measures for the given coordination geometry for each permutation investigated.
        """
    permutations_symmetry_measures = [None] * NRANDOM
    permutations = []
    algos = []
    perfect2local_maps = []
    local2perfect_maps = []
    for idx in range(NRANDOM):
        perm = np.random.permutation(coordination_geometry.coordination_number)
        permutations.append(perm)
        p2l = {}
        l2p = {}
        for i_p, pp in enumerate(perm):
            p2l[i_p] = pp
            l2p[pp] = i_p
        perfect2local_maps.append(p2l)
        local2perfect_maps.append(l2p)
        points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
        sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
        sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
        permutations_symmetry_measures[idx] = sm_info
        algos.append('APPROXIMATE_FALLBACK')
    return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)