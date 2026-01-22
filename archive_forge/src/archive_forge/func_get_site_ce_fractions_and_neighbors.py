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
def get_site_ce_fractions_and_neighbors(self, site, full_ce_info=False, strategy_info=False):
    """
        Applies the strategy to the structure_environments object in order to get coordination environments, their
        fraction, csm, geometry_info, and neighbors

        Args:
            site: Site for which the above information is sought

        Returns:
            The list of neighbors of the site. For complex strategies, where one allows multiple solutions, this
        can return a list of list of neighbors.
        """
    isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
    geoms_and_maps_list = self.get_site_coordination_environments_fractions(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_maps=True, return_strategy_dict_info=True)
    if geoms_and_maps_list is None:
        return None
    site_nbs_sets = self.structure_environments.neighbors_sets[isite]
    ce_and_neighbors = []
    for fractions_dict in geoms_and_maps_list:
        ce_map = fractions_dict['ce_map']
        ce_nb_set = site_nbs_sets[ce_map[0]][ce_map[1]]
        neighbors = [{'site': nb_site_and_index['site'], 'index': nb_site_and_index['index']} for nb_site_and_index in ce_nb_set.neighb_sites_and_indices]
        fractions_dict['neighbors'] = neighbors
        ce_and_neighbors.append(fractions_dict)
    return ce_and_neighbors