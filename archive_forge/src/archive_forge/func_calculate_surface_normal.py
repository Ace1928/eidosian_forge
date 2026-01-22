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
def calculate_surface_normal() -> np.ndarray:
    """Calculate the unit surface normal vector using the reciprocal
            lattice vector.
            """
    recip_lattice = lattice.reciprocal_lattice_crystallographic
    normal = recip_lattice.get_cartesian_coords(miller_index)
    normal /= np.linalg.norm(normal)
    return normal