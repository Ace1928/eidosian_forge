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
def get_slab_regions(slab: Slab, blength: float=3.5) -> list[tuple[float, float]]:
    """Find the z-ranges for the slab region.

    Useful for discerning where the slab ends and vacuum begins
    if the slab is not fully within the cell.

    TODO (@DanielYang59): this should be a method for `Slab`?

    TODO (@DanielYang59): maybe project all z coordinates to 1D?

    Args:
        slab (Slab): The Slab to analyse.
        blength (float): The bond length between atoms in Angstrom.
            You generally want this value to be larger than the actual
            bond length in order to find atoms that are part of the slab.
    """
    frac_coords: list = []
    indices: list = []
    all_indices: list = []
    for site in slab:
        neighbors = slab.get_neighbors(site, blength)
        for nn in neighbors:
            if nn[0].frac_coords[2] < 0:
                frac_coords.append(nn[0].frac_coords[2])
                indices.append(nn[-2])
                if nn[-2] not in all_indices:
                    all_indices.append(nn[-2])
    if frac_coords:
        while frac_coords:
            last_fcoords = copy.copy(frac_coords)
            last_indices = copy.copy(indices)
            site = slab[indices[frac_coords.index(min(frac_coords))]]
            neighbors = slab.get_neighbors(site, blength, include_index=True, include_image=True)
            frac_coords, indices = ([], [])
            for nn in neighbors:
                if 1 > nn[0].frac_coords[2] > 0 and nn[0].frac_coords[2] < site.frac_coords[2]:
                    frac_coords.append(nn[0].frac_coords[2])
                    indices.append(nn[-2])
                    if nn[-2] not in all_indices:
                        all_indices.append(nn[-2])
        upper_fcoords: list = []
        for site in slab:
            if all((nn.index not in all_indices for nn in slab.get_neighbors(site, blength))):
                upper_fcoords.append(site.frac_coords[2])
        coords: list = copy.copy(frac_coords) if frac_coords else copy.copy(last_fcoords)
        min_top = slab[last_indices[coords.index(min(coords))]].frac_coords[2]
        return [(0, max(upper_fcoords)), (min_top, 1)]
    sorted_sites = sorted(slab, key=lambda site: site.frac_coords[2])
    return [(sorted_sites[0].frac_coords[2], sorted_sites[-1].frac_coords[2])]