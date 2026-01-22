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
def gen_possible_shifts(ftol: float) -> list[float]:
    """Generate possible shifts by clustering z coordinates.

            Args:
                ftol (float): Threshold for fcluster to check if
                    two atoms are on the same plane.
            """
    frac_coords = self.oriented_unit_cell.frac_coords
    n_atoms = len(frac_coords)
    if n_atoms == 1:
        shift = frac_coords[0][2] + 0.5
        return [shift - math.floor(shift)]
    dist_matrix = np.zeros((n_atoms, n_atoms))
    for i, j in itertools.combinations(list(range(n_atoms)), 2):
        if i != j:
            z_dist = frac_coords[i][2] - frac_coords[j][2]
            z_dist = abs(z_dist - round(z_dist)) * self._proj_height
            dist_matrix[i, j] = z_dist
            dist_matrix[j, i] = z_dist
    z_matrix = linkage(squareform(dist_matrix))
    clusters = fcluster(z_matrix, ftol, criterion='distance')
    clst_loc = {c: frac_coords[i][2] for i, c in enumerate(clusters)}
    possible_clst = [coord - math.floor(coord) for coord in sorted(clst_loc.values())]
    n_shifts = len(possible_clst)
    shifts = []
    for i in range(n_shifts):
        if i == n_shifts - 1:
            shift = (possible_clst[0] + 1 + possible_clst[i]) * 0.5
        else:
            shift = (possible_clst[i] + possible_clst[i + 1]) * 0.5
        shifts.append(shift - math.floor(shift))
    return sorted(shifts)