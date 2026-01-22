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
def to_bson_voronoi_list2(self):
    """
        Transforms the voronoi_list into a vlist + bson_nb_voro_list, that are BSON-encodable.

        Returns:
            [vlist, bson_nb_voro_list], to be used in the as_dict method.
        """
    bson_nb_voro_list2 = [None] * len(self.voronoi_list2)
    for ivoro, voro in enumerate(self.voronoi_list2):
        if voro is None or voro == 'None':
            continue
        site_voro = []
        for nb_dict in voro:
            site = nb_dict['site']
            site_dict = {key: val for key, val in nb_dict.items() if key != 'site'}
            diff = site.frac_coords - self.structure[nb_dict['index']].frac_coords
            site_voro.append([[nb_dict['index'], [float(c) for c in diff]], site_dict])
        bson_nb_voro_list2[ivoro] = site_voro
    return bson_nb_voro_list2