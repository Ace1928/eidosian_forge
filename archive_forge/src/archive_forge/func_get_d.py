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
def get_d(slab: Slab) -> float:
    """Determine the z-spacing between the bottom two layers for a Slab.

    TODO (@DanielYang59): this should be private/internal to ReconstructionGenerator?
    """
    sorted_sites = sorted(slab, key=lambda site: site.frac_coords[2])
    for site, next_site in zip(sorted_sites, sorted_sites[1:]):
        if not isclose(site.frac_coords[2], next_site.frac_coords[2], abs_tol=1e-06):
            distance = next_site.frac_coords[2] - site.frac_coords[2]
            break
    return slab.lattice.get_cartesian_coords([0, 0, distance])[2]