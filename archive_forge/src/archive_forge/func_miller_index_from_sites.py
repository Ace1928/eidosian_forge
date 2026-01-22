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
def miller_index_from_sites(lattice: Lattice | ArrayLike, coords: ArrayLike, coords_are_cartesian: bool=True, round_dp: int=4, verbose: bool=True) -> tuple[int, int, int]:
    """Get the Miller index of a plane, determined by a given set of coordinates.

    A minimum of 3 sets of coordinates are required. If more than 3
    coordinates are given, the plane that minimises the distance to all
    sites will be calculated.

    Args:
        lattice (matrix or Lattice): A 3x3 lattice matrix or `Lattice` object.
        coords (ArrayLike): A list or numpy array of coordinates. Can be
            Cartesian or fractional coordinates.
        coords_are_cartesian (bool, optional): Whether the coordinates are
            in Cartesian coordinates, or fractional (False).
        round_dp (int, optional): The number of decimal places to round the
            Miller index to.
        verbose (bool, optional): Whether to print warnings.

    Returns:
        tuple[int]: The Miller index.
    """
    if not isinstance(lattice, Lattice):
        lattice = Lattice(lattice)
    return lattice.get_miller_index_from_coords(coords, coords_are_cartesian=coords_are_cartesian, round_dp=round_dp, verbose=verbose)