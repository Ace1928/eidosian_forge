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
def coordination_geometry_symmetry_measures_standard(self, coordination_geometry, algo, points_perfect=None, optimization=None):
    """
        Returns the symmetry measures for a set of permutations (whose setup depends on the coordination geometry)
        for the coordination geometry "coordination_geometry". Standard implementation looking for the symmetry
        measures of each permutation

        Args:
            coordination_geometry: The coordination geometry to be investigated

        Returns:
            The symmetry measures for the given coordination geometry for each permutation investigated.
        """
    if optimization == 2:
        permutations_symmetry_measures = [None] * len(algo.permutations)
        permutations = []
        algos = []
        local2perfect_maps = []
        perfect2local_maps = []
        for idx, perm in enumerate(algo.permutations):
            local2perfect_map = {}
            perfect2local_map = {}
            permutations.append(perm)
            for iperfect, ii in enumerate(perm):
                perfect2local_map[iperfect] = ii
                local2perfect_map[ii] = iperfect
            local2perfect_maps.append(local2perfect_map)
            perfect2local_maps.append(perfect2local_map)
            points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
            sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
            sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
            permutations_symmetry_measures[idx] = sm_info
            algos.append(str(algo))
        return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)
    permutations_symmetry_measures = [None] * len(algo.permutations)
    permutations = []
    algos = []
    local2perfect_maps = []
    perfect2local_maps = []
    for idx, perm in enumerate(algo.permutations):
        local2perfect_map = {}
        perfect2local_map = {}
        permutations.append(perm)
        for iperfect, ii in enumerate(perm):
            perfect2local_map[iperfect] = ii
            local2perfect_map[ii] = iperfect
        local2perfect_maps.append(local2perfect_map)
        perfect2local_maps.append(perfect2local_map)
        points_distorted = self.local_geometry.points_wcs_ctwcc(permutation=perm)
        sm_info = symmetry_measure(points_distorted=points_distorted, points_perfect=points_perfect)
        sm_info['translation_vector'] = self.local_geometry.centroid_with_centre
        permutations_symmetry_measures[idx] = sm_info
        algos.append(str(algo))
    return (permutations_symmetry_measures, permutations, algos, local2perfect_maps, perfect2local_maps)