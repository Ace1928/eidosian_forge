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
def hkl_transformation(transf: np.ndarray, miller_index: tuple[int, int, int]) -> tuple[int, int, int]:
    """Transform the Miller index from setting A to B with a transformation matrix.

    Args:
        transf (3x3 array): The matrix that transforms a lattice from A to B.
        miller_index (tuple[int, int, int]): The Miller index [h, k, l] to transform.
    """

    def math_lcm(a: int, b: int) -> int:
        """Calculate the least common multiple."""
        return a * b // math.gcd(a, b)
    reduced_transf = reduce(math_lcm, [int(1 / i) for i in itertools.chain(*transf) if i != 0]) * transf
    reduced_transf = reduced_transf.astype(int)
    transf_hkl = np.dot(reduced_transf, miller_index)
    divisor = abs(reduce(gcd, transf_hkl))
    transf_hkl = np.array([idx // divisor for idx in transf_hkl])
    if len([i for i in transf_hkl if i < 0]) > 1:
        transf_hkl *= -1
    return tuple(transf_hkl)