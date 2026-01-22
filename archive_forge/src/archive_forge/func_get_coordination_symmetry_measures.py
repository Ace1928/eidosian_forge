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
def get_coordination_symmetry_measures(self, only_minimum=True, all_csms=True, optimization=None):
    """
        Returns the continuous symmetry measures of the current local geometry in a dictionary.

        Returns:
            the continuous symmetry measures of the current local geometry in a dictionary.
        """
    test_geometries = self.allcg.get_implemented_geometries(len(self.local_geometry.coords))
    if len(self.local_geometry.coords) == 1:
        if len(test_geometries) == 0:
            return {}
        result_dict = {'S:1': {'csm': 0.0, 'indices': [0], 'algo': 'EXPLICIT', 'local2perfect_map': {0: 0}, 'perfect2local_map': {0: 0}, 'scaling_factor': None, 'rotation_matrix': None, 'translation_vector': None}}
        if all_csms:
            for csmtype in ['wocs_ctwocc', 'wocs_ctwcc', 'wocs_csc', 'wcs_ctwocc', 'wcs_ctwcc', 'wcs_csc']:
                result_dict['S:1'][f'csm_{csmtype}'] = 0.0
                result_dict['S:1'][f'scaling_factor_{csmtype}'] = None
                result_dict['S:1'][f'rotation_matrix_{csmtype}'] = None
                result_dict['S:1'][f'translation_vector_{csmtype}'] = None
        return result_dict
    result_dict = {}
    for geometry in test_geometries:
        self.perfect_geometry = AbstractGeometry.from_cg(cg=geometry, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid)
        points_perfect = self.perfect_geometry.points_wcs_ctwcc()
        cgsm = self.coordination_geometry_symmetry_measures(geometry, points_perfect=points_perfect, optimization=optimization)
        result, permutations, algos, local2perfect_maps, perfect2local_maps = cgsm
        if only_minimum:
            if len(result) > 0:
                imin = np.argmin([rr['symmetry_measure'] for rr in result])
                algo = algos[imin] if geometry.algorithms is not None else algos
                result_dict[geometry.mp_symbol] = {'csm': result[imin]['symmetry_measure'], 'indices': permutations[imin], 'algo': algo, 'local2perfect_map': local2perfect_maps[imin], 'perfect2local_map': perfect2local_maps[imin], 'scaling_factor': 1.0 / result[imin]['scaling_factor'], 'rotation_matrix': np.linalg.inv(result[imin]['rotation_matrix']), 'translation_vector': result[imin]['translation_vector']}
                if all_csms:
                    self._update_results_all_csms(result_dict, permutations, imin, geometry)
        else:
            result_dict[geometry.mp_symbol] = {'csm': result, 'indices': permutations, 'algo': algos, 'local2perfect_map': local2perfect_maps, 'perfect2local_map': perfect2local_maps}
    return result_dict