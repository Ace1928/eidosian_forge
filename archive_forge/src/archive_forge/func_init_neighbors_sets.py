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
def init_neighbors_sets(self, isite, additional_conditions=None, valences=None):
    """
        Initialize the list of neighbors sets for the current site.

        Args:
            isite: Index of the site under consideration.
            additional_conditions: Additional conditions to be used for the initialization of the list of
                neighbors sets, e.g. "Only anion-cation bonds", ...
            valences: List of valences for each site in the structure (needed if an additional condition based on the
                valence is used, e.g. only anion-cation bonds).
        """
    site_voronoi = self.voronoi.voronoi_list2[isite]
    if site_voronoi is None:
        return
    if additional_conditions is None:
        additional_conditions = self.AC.ALL
    if (self.AC.ONLY_ACB in additional_conditions or self.AC.ONLY_ACB_AND_NO_E2SEB) and valences is None:
        raise ChemenvError('StructureEnvironments', 'init_neighbors_sets', 'Valences are not given while only_anion_cation_bonds are allowed. Cannot continue')
    site_distance_parameters = self.voronoi.neighbors_normalized_distances[isite]
    site_angle_parameters = self.voronoi.neighbors_normalized_angles[isite]
    distance_conditions = []
    for idp, dp_dict in enumerate(site_distance_parameters):
        distance_conditions.append([])
        for inb in range(len(site_voronoi)):
            cond = inb in dp_dict['nb_indices']
            distance_conditions[idp].append(cond)
    angle_conditions = []
    for iap, ap_dict in enumerate(site_angle_parameters):
        angle_conditions.append([])
        for inb in range(len(site_voronoi)):
            cond = inb in ap_dict['nb_indices']
            angle_conditions[iap].append(cond)
    precomputed_additional_conditions = {ac: [] for ac in additional_conditions}
    for voro_nb_dict in site_voronoi:
        for ac in additional_conditions:
            cond = self.AC.check_condition(condition=ac, structure=self.structure, parameters={'valences': valences, 'neighbor_index': voro_nb_dict['index'], 'site_index': isite})
            precomputed_additional_conditions[ac].append(cond)
    for idp, dp_dict in enumerate(site_distance_parameters):
        for iap, ap_dict in enumerate(site_angle_parameters):
            for iac, ac in enumerate(additional_conditions):
                src = {'origin': 'dist_ang_ac_voronoi', 'idp': idp, 'iap': iap, 'dp_dict': dp_dict, 'ap_dict': ap_dict, 'iac': iac, 'ac': ac, 'ac_name': self.AC.CONDITION_DESCRIPTION[ac]}
                site_voronoi_indices = [inb for inb, _voro_nb_dict in enumerate(site_voronoi) if distance_conditions[idp][inb] and angle_conditions[iap][inb] and precomputed_additional_conditions[ac][inb]]
                nb_set = self.NeighborsSet(structure=self.structure, isite=isite, detailed_voronoi=self.voronoi, site_voronoi_indices=site_voronoi_indices, sources=src)
                self.add_neighbors_set(isite=isite, nb_set=nb_set)