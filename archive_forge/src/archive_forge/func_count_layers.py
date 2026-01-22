from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def count_layers(struct: Structure, el=None) -> int:
    """Counts the number of 'layers' along the c-axis."""
    el = el or struct.elements[0]
    frac_coords = [site.frac_coords for site in struct if site.species_string == str(el)]
    n = len(frac_coords)
    if n == 1:
        return 1
    dist_matrix = np.zeros((n, n))
    h = struct.lattice.c
    for ii, jj in combinations(list(range(n)), 2):
        if ii != jj:
            cdist = frac_coords[ii][2] - frac_coords[jj][2]
            cdist = abs(cdist - round(cdist)) * h
            dist_matrix[ii, jj] = cdist
            dist_matrix[jj, ii] = cdist
    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, 0.25, criterion='distance')
    clustered_sites: dict[int, list[Site]] = {c: [] for c in clusters}
    for idx, cluster in enumerate(clusters):
        clustered_sites[cluster].append(struct[idx])
    plane_heights = {np.average(np.mod([s.frac_coords[2] for s in sites], 1)): c for c, sites in clustered_sites.items()}
    return len(plane_heights)