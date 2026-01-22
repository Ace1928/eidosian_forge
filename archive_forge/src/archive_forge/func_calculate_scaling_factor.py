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
def calculate_scaling_factor() -> np.ndarray:
    """Calculate scaling factor.

            # TODO (@DanielYang59): revise docstring to add more details.
            """
    slab_scale_factor = []
    non_orth_ind = []
    eye = np.eye(3, dtype=int)
    for idx, miller_idx in enumerate(miller_index):
        if miller_idx == 0:
            slab_scale_factor.append(eye[idx])
        else:
            d = abs(np.dot(normal, lattice.matrix[idx])) / lattice.abc[idx]
            non_orth_ind.append((idx, d))
    c_index, _dist = max(non_orth_ind, key=lambda t: t[1])
    if len(non_orth_ind) > 1:
        lcm_miller = lcm(*(miller_index[i] for i, _d in non_orth_ind))
        for (ii, _di), (jj, _dj) in itertools.combinations(non_orth_ind, 2):
            scale_factor = [0, 0, 0]
            scale_factor[ii] = -int(round(lcm_miller / miller_index[ii]))
            scale_factor[jj] = int(round(lcm_miller / miller_index[jj]))
            slab_scale_factor.append(scale_factor)
            if len(slab_scale_factor) == 2:
                break
    if max_normal_search is None:
        slab_scale_factor.append(eye[c_index])
    else:
        index_range = sorted(range(-max_normal_search, max_normal_search + 1), key=lambda x: -abs(x))
        candidates = []
        for uvw in itertools.product(index_range, index_range, index_range):
            if not any(uvw) or abs(np.linalg.det([*slab_scale_factor, uvw])) < 1e-08:
                continue
            vec = lattice.get_cartesian_coords(uvw)
            osdm = np.linalg.norm(vec)
            cosine = abs(np.dot(vec, normal) / osdm)
            candidates.append((uvw, cosine, osdm))
            if isclose(abs(cosine), 1, abs_tol=1e-08):
                break
        uvw, cosine, osdm = max(candidates, key=lambda x: (x[1], -x[2]))
        slab_scale_factor.append(uvw)
    slab_scale_factor = np.array(slab_scale_factor)
    if np.linalg.det(slab_scale_factor) < 0:
        slab_scale_factor *= -1
    reduced_scale_factor = [reduce_vector(v) for v in slab_scale_factor]
    return np.array(reduced_scale_factor)