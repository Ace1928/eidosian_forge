from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def center_slab(slab: Slab) -> Slab:
    """Relocate the Slab to the center such that its center
    (the slab region) is close to z=0.5.

    This makes it easier to find surface sites and apply
    operations like doping.

    There are two possible cases:

        1. When the slab region is completely positioned between
        two vacuum layers in the cell but is not centered, we simply
        shift the Slab to the center along z-axis.

        2. If the Slab completely resides outside the cell either
        from the bottom or the top, we iterate through all sites that
        spill over and shift all sites such that it is now
        on the other side. An edge case being, either the top
        of the Slab is at z = 0 or the bottom is at z = 1.

    TODO (@DanielYang59): this should be a method for `Slab`?

    Args:
        slab (Slab): The Slab to center.

    Returns:
        Slab: The centered Slab.
    """
    all_indices = list(range(len(slab)))
    bond_dists = sorted((nn[1] for nn in slab.get_neighbors(slab[0], 10) if nn[1] > 0))
    cutoff_radius = bond_dists[0] * 3
    for site in slab:
        if any((nn[1] >= slab.lattice.c for nn in slab.get_neighbors(site, cutoff_radius))):
            shift = 1 - site.frac_coords[2] + 0.05
            slab.translate_sites(all_indices, [0, 0, shift])
    weights = [site.species.weight for site in slab]
    center_of_mass = np.average(slab.frac_coords, weights=weights, axis=0)
    shift = 0.5 - center_of_mass[2]
    slab.translate_sites(all_indices, [0, 0, shift])
    return slab