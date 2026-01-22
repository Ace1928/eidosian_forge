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
def get_site_info_for_specie_allces(self, specie, min_fraction=0):
    """
        Get list of indices that have the given specie.

        Args:
            specie: Species to get.
            min_fraction: Minimum fraction of the coordination environment.

        Returns:
            dict: with the list of coordination environments for the given species, the indices of the sites
                in which they appear, their fractions and continuous symmetry measures.
        """
    allces = {}
    element = specie.symbol
    oxi_state = specie.oxi_state
    for isite, site in enumerate(self.structure):
        if element in [sp.symbol for sp in site.species] and self.valences == 'undefined' or oxi_state == self.valences[isite]:
            if self.coordination_environments[isite] is None:
                continue
            for ce_dict in self.coordination_environments[isite]:
                if ce_dict['ce_fraction'] < min_fraction:
                    continue
                if ce_dict['ce_symbol'] not in allces:
                    allces[ce_dict['ce_symbol']] = {'isites': [], 'fractions': [], 'csms': []}
                allces[ce_dict['ce_symbol']]['isites'].append(isite)
                allces[ce_dict['ce_symbol']]['fractions'].append(ce_dict['ce_fraction'])
                allces[ce_dict['ce_symbol']]['csms'].append(ce_dict['csm'])
    return allces