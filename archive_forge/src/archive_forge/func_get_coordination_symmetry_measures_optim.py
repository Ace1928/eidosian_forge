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
def get_coordination_symmetry_measures_optim(self, only_minimum=True, all_csms=True, nb_set=None, optimization=None):
    """
        Returns the continuous symmetry measures of the current local geometry in a dictionary.

        Returns:
            the continuous symmetry measures of the current local geometry in a dictionary.
        """
    cn = len(self.local_geometry.coords)
    test_geometries = self.allcg.get_implemented_geometries(cn)
    if all((cg.algorithms[0].algorithm_type == EXPLICIT_PERMUTATIONS for cg in test_geometries)):
        return self.get_coordination_symmetry_measures(only_minimum=only_minimum, all_csms=all_csms, optimization=optimization)
    if not all((all((algo.algorithm_type == SEPARATION_PLANE for algo in cg.algorithms)) for cg in test_geometries)):
        raise ValueError('All algorithms should be EXPLICIT_PERMUTATIONS or SEPARATION_PLANE')
    result_dict = {}
    for geometry in test_geometries:
        logging.log(level=5, msg=f'Getting Continuous Symmetry Measure with Separation Plane algorithm for geometry "{geometry.ce_symbol}"')
        self.perfect_geometry = AbstractGeometry.from_cg(cg=geometry, centering_type=self.centering_type, include_central_site_in_centroid=self.include_central_site_in_centroid)
        points_perfect = self.perfect_geometry.points_wcs_ctwcc()
        cgsm = self.coordination_geometry_symmetry_measures_sepplane_optim(geometry, points_perfect=points_perfect, nb_set=nb_set, optimization=optimization)
        result, permutations, algos, local2perfect_maps, perfect2local_maps = cgsm
        if only_minimum and len(result) > 0:
            imin = np.argmin([rr['symmetry_measure'] for rr in result])
            algo = algos[imin] if geometry.algorithms is not None else algos
            result_dict[geometry.mp_symbol] = {'csm': result[imin]['symmetry_measure'], 'indices': permutations[imin], 'algo': algo, 'local2perfect_map': local2perfect_maps[imin], 'perfect2local_map': perfect2local_maps[imin], 'scaling_factor': 1.0 / result[imin]['scaling_factor'], 'rotation_matrix': np.linalg.inv(result[imin]['rotation_matrix']), 'translation_vector': result[imin]['translation_vector']}
            if all_csms:
                self._update_results_all_csms(result_dict, permutations, imin, geometry)
    return result_dict