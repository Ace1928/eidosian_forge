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
def get_symmetrically_equivalent_miller_indices(structure: Structure, miller_index: tuple[int, ...], return_hkil: bool=True, system: CrystalSystem | None=None) -> list:
    """Get indices for all equivalent sites within a given structure.
    Analysis is based on the symmetry of its reciprocal lattice.

    Args:
        structure (Structure): Structure to analyze.
        miller_index (tuple): Designates the family of Miller indices
            to find. Can be hkl or hkil for hexagonal systems.
        return_hkil (bool): Whether to return hkil (True) form of Miller
            index for hexagonal systems, or hkl (False).
        system: The crystal system of the structure.
    """
    if len(miller_index) >= 3:
        _miller_index: tuple[int, int, int] = (miller_index[0], miller_index[1], miller_index[-1])
    max_idx = max(np.abs(miller_index))
    idx_range = list(range(-max_idx, max_idx + 1))
    idx_range.reverse()
    if system:
        spg_analyzer = None
    else:
        spg_analyzer = SpacegroupAnalyzer(structure)
        system = spg_analyzer.get_crystal_system()
    if system == 'trigonal':
        if not spg_analyzer:
            spg_analyzer = SpacegroupAnalyzer(structure)
        prim_structure = spg_analyzer.get_primitive_standard_structure()
        symm_ops = prim_structure.lattice.get_recp_symmetry_operation()
    else:
        symm_ops = structure.lattice.get_recp_symmetry_operation()
    equivalent_millers: list[tuple[int, int, int]] = [_miller_index]
    for miller in itertools.product(idx_range, idx_range, idx_range):
        if miller == _miller_index:
            continue
        if any((idx != 0 for idx in miller)):
            if _is_in_miller_family(miller, equivalent_millers, symm_ops):
                equivalent_millers += [miller]
            if all((max_idx > i for i in np.abs(miller))) and (not in_coord_list(equivalent_millers, miller)) and _is_in_miller_family(max_idx * np.array(miller), equivalent_millers, symm_ops):
                equivalent_millers += [miller]
    if return_hkil and system in {'trigonal', 'hexagonal'}:
        return [(hkl[0], hkl[1], -1 * hkl[0] - hkl[1], hkl[2]) for hkl in equivalent_millers]
    return equivalent_millers