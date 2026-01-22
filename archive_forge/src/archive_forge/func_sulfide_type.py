from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def sulfide_type(structure):
    """
    Determines if a structure is a sulfide/polysulfide/sulfate.

    Args:
        structure (Structure): Input structure.

    Returns:
        str: sulfide/polysulfide or None if structure is a sulfate.
    """
    structure = structure.copy().remove_oxidation_states()
    sulphur = Element('S')
    comp = structure.composition
    if comp.is_element or sulphur not in comp:
        return None
    try:
        finder = SpacegroupAnalyzer(structure, symprec=0.1)
        symm_structure = finder.get_symmetrized_structure()
        s_sites = [sites[0] for sites in symm_structure.equivalent_sites if sites[0].specie == sulphur]
    except Exception:
        s_sites = [site for site in structure if site.specie == sulphur]

    def process_site(site):
        search_radius = 4
        neighbors = []
        while len(neighbors) == 0:
            neighbors = structure.get_neighbors(site, search_radius)
            search_radius *= 2
            if search_radius > max(structure.lattice.abc) * 2:
                break
        neighbors = sorted(neighbors, key=lambda n: n.nn_distance)
        dist = neighbors[0].nn_distance
        coord_elements = [nn.specie for nn in neighbors if nn.nn_distance < dist + 0.4][:4]
        avg_electroneg = np.mean([elem.X for elem in coord_elements])
        if avg_electroneg > sulphur.X:
            return 'sulfate'
        if avg_electroneg == sulphur.X and sulphur in coord_elements:
            return 'polysulfide'
        return 'sulfide'
    types = {process_site(site) for site in s_sites}
    if 'sulfate' in types:
        return None
    if 'polysulfide' in types:
        return 'polysulfide'
    return 'sulfide'