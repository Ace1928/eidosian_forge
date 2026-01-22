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
def differences_wrt(self, other):
    """
        Return differences found in the current StructureEnvironments with respect to another StructureEnvironments.

        Args:
            other: A StructureEnvironments object.

        Returns:
            List of differences between the two StructureEnvironments objects.
        """
    differences = []
    if self.structure != other.structure:
        differences += ({'difference': 'structure', 'comparison': '__eq__', 'self': self.structure, 'other': other.structure}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
        return differences
    if self.valences != other.valences:
        differences.append({'difference': 'valences', 'comparison': '__eq__', 'self': self.valences, 'other': other.valences})
    if self.info != other.info:
        differences.append({'difference': 'info', 'comparison': '__eq__', 'self': self.info, 'other': other.info})
    if self.voronoi != other.voronoi:
        if self.voronoi.is_close_to(other.voronoi):
            differences += ({'difference': 'voronoi', 'comparison': '__eq__', 'self': self.voronoi, 'other': other.voronoi}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
            return differences
        differences += ({'difference': 'voronoi', 'comparison': 'is_close_to', 'self': self.voronoi, 'other': other.voronoi}, {'difference': 'PREVIOUS DIFFERENCE IS DISMISSIVE', 'comparison': 'differences_wrt'})
        return differences
    for isite, self_site_nb_sets in enumerate(self.neighbors_sets):
        other_site_nb_sets = other.neighbors_sets[isite]
        if self_site_nb_sets is None:
            if other_site_nb_sets is None:
                continue
            differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'has_neighbors', 'self': 'None', 'other': set(other_site_nb_sets)})
            continue
        if other_site_nb_sets is None:
            differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'has_neighbors', 'self': set(self_site_nb_sets), 'other': 'None'})
            continue
        self_site_cns = set(self_site_nb_sets)
        other_site_cns = set(other_site_nb_sets)
        if self_site_cns != other_site_cns:
            differences.append({'difference': f'neighbors_sets[isite={isite!r}]', 'comparison': 'coordination_numbers', 'self': self_site_cns, 'other': other_site_cns})
        common_cns = self_site_cns.intersection(other_site_cns)
        for cn in common_cns:
            other_site_cn_nb_sets = other_site_nb_sets[cn]
            self_site_cn_nb_sets = self_site_nb_sets[cn]
            set_self_site_cn_nb_sets = set(self_site_cn_nb_sets)
            set_other_site_cn_nb_sets = set(other_site_cn_nb_sets)
            if set_self_site_cn_nb_sets != set_other_site_cn_nb_sets:
                differences.append({'difference': f'neighbors_sets[isite={isite!r}][cn={cn!r}]', 'comparison': 'neighbors_sets', 'self': self_site_cn_nb_sets, 'other': other_site_cn_nb_sets})
            common_nb_sets = set_self_site_cn_nb_sets.intersection(set_other_site_cn_nb_sets)
            for nb_set in common_nb_sets:
                inb_set_self = self_site_cn_nb_sets.index(nb_set)
                inb_set_other = other_site_cn_nb_sets.index(nb_set)
                self_ce = self.ce_list[isite][cn][inb_set_self]
                other_ce = other.ce_list[isite][cn][inb_set_other]
                if self_ce != other_ce:
                    if self_ce.is_close_to(other_ce):
                        differences.append({'difference': f'ce_list[isite={isite!r}][cn={cn!r}][inb_set={inb_set_self}]', 'comparison': '__eq__', 'self': self_ce, 'other': other_ce})
                    else:
                        differences.append({'difference': f'ce_list[isite={isite!r}][cn={cn!r}][inb_set={inb_set_self}]', 'comparison': 'is_close_to', 'self': self_ce, 'other': other_ce})
    return differences