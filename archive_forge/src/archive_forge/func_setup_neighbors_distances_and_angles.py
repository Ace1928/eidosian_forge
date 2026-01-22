from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
def setup_neighbors_distances_and_angles(self, indices):
    """
        Initializes the angle and distance separations.

        Args:
            indices: Indices of the sites for which the Voronoi is needed.
        """
    self.neighbors_distances = [None] * len(self.structure)
    self.neighbors_normalized_distances = [None] * len(self.structure)
    self.neighbors_angles = [None] * len(self.structure)
    self.neighbors_normalized_angles = [None] * len(self.structure)
    for isite in indices:
        results = self.voronoi_list2[isite]
        if results is None:
            continue
        self.neighbors_distances[isite] = []
        self.neighbors_normalized_distances[isite] = []
        normalized_distances = [nb_dict['normalized_distance'] for nb_dict in results]
        isorted_distances = np.argsort(normalized_distances)
        self.neighbors_normalized_distances[isite].append({'min': normalized_distances[isorted_distances[0]], 'max': normalized_distances[isorted_distances[0]]})
        self.neighbors_distances[isite].append({'min': results[isorted_distances[0]]['distance'], 'max': results[isorted_distances[0]]['distance']})
        icurrent = 0
        nb_indices = {int(isorted_distances[0])}
        dnb_indices = {int(isorted_distances[0])}
        for idist in iter(isorted_distances):
            wd = normalized_distances[idist]
            if self.maximum_distance_factor is not None and wd > self.maximum_distance_factor:
                self.neighbors_normalized_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_normalized_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                self.neighbors_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                break
            if np.isclose(wd, self.neighbors_normalized_distances[isite][icurrent]['max'], rtol=0.0, atol=self.normalized_distance_tolerance):
                self.neighbors_normalized_distances[isite][icurrent]['max'] = wd
                self.neighbors_distances[isite][icurrent]['max'] = results[idist]['distance']
                dnb_indices.add(int(idist))
            else:
                self.neighbors_normalized_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_normalized_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                self.neighbors_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                dnb_indices = {int(idist)}
                self.neighbors_normalized_distances[isite].append({'min': wd, 'max': wd})
                self.neighbors_distances[isite].append({'min': results[idist]['distance'], 'max': results[idist]['distance']})
                icurrent += 1
            nb_indices.add(int(idist))
        else:
            self.neighbors_normalized_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
            self.neighbors_distances[isite][icurrent]['nb_indices'] = list(nb_indices)
            self.neighbors_normalized_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
            self.neighbors_distances[isite][icurrent]['dnb_indices'] = list(dnb_indices)
        for idist in range(len(self.neighbors_distances[isite]) - 1):
            dist_dict = self.neighbors_distances[isite][idist]
            dist_dict_next = self.neighbors_distances[isite][idist + 1]
            dist_dict['next'] = dist_dict_next['min']
            ndist_dict = self.neighbors_normalized_distances[isite][idist]
            ndist_dict_next = self.neighbors_normalized_distances[isite][idist + 1]
            ndist_dict['next'] = ndist_dict_next['min']
        if self.maximum_distance_factor is not None:
            dfact = self.maximum_distance_factor
        else:
            dfact = self.default_voronoi_cutoff / self.neighbors_distances[isite][0]['min']
        self.neighbors_normalized_distances[isite][-1]['next'] = dfact
        self.neighbors_distances[isite][-1]['next'] = dfact * self.neighbors_distances[isite][0]['min']
        self.neighbors_angles[isite] = []
        self.neighbors_normalized_angles[isite] = []
        normalized_angles = [nb_dict['normalized_angle'] for nb_dict in results]
        isorted_angles = np.argsort(normalized_angles)[::-1]
        self.neighbors_normalized_angles[isite].append({'max': normalized_angles[isorted_angles[0]], 'min': normalized_angles[isorted_angles[0]]})
        self.neighbors_angles[isite].append({'max': results[isorted_angles[0]]['angle'], 'min': results[isorted_angles[0]]['angle']})
        icurrent = 0
        nb_indices = {int(isorted_angles[0])}
        dnb_indices = {int(isorted_angles[0])}
        for iang in iter(isorted_angles):
            wa = normalized_angles[iang]
            if self.minimum_angle_factor is not None and wa < self.minimum_angle_factor:
                self.neighbors_normalized_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_normalized_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                self.neighbors_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                break
            if np.isclose(wa, self.neighbors_normalized_angles[isite][icurrent]['min'], rtol=0.0, atol=self.normalized_angle_tolerance):
                self.neighbors_normalized_angles[isite][icurrent]['min'] = wa
                self.neighbors_angles[isite][icurrent]['min'] = results[iang]['angle']
                dnb_indices.add(int(iang))
            else:
                self.neighbors_normalized_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
                self.neighbors_normalized_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                self.neighbors_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
                dnb_indices = {int(iang)}
                self.neighbors_normalized_angles[isite].append({'max': wa, 'min': wa})
                self.neighbors_angles[isite].append({'max': results[iang]['angle'], 'min': results[iang]['angle']})
                icurrent += 1
            nb_indices.add(int(iang))
        else:
            self.neighbors_normalized_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
            self.neighbors_angles[isite][icurrent]['nb_indices'] = list(nb_indices)
            self.neighbors_normalized_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
            self.neighbors_angles[isite][icurrent]['dnb_indices'] = list(dnb_indices)
        for iang in range(len(self.neighbors_angles[isite]) - 1):
            ang_dict = self.neighbors_angles[isite][iang]
            ang_dict_next = self.neighbors_angles[isite][iang + 1]
            ang_dict['next'] = ang_dict_next['max']
            nang_dict = self.neighbors_normalized_angles[isite][iang]
            nang_dict_next = self.neighbors_normalized_angles[isite][iang + 1]
            nang_dict['next'] = nang_dict_next['max']
        afact = self.minimum_angle_factor if self.minimum_angle_factor is not None else 0.0
        self.neighbors_normalized_angles[isite][-1]['next'] = afact
        self.neighbors_angles[isite][-1]['next'] = afact * self.neighbors_angles[isite][0]['max']