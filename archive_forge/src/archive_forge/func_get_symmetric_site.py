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
def get_symmetric_site(self, point: ArrayLike, cartesian: bool=False) -> ArrayLike:
    """This method uses symmetry operations to find an equivalent site on
        the other side of the slab. Works mainly for slabs with Laue symmetry.

        This is useful for retaining the non-polar and
        symmetric properties of a slab when creating adsorbed
        structures or symmetric reconstructions.

        TODO (@DanielYang59): use "site" over "point" as arg name for consistency

        Args:
            point (ArrayLike): Fractional coordinate of the original site.
            cartesian (bool): Use Cartesian coordinates.

        Returns:
            ArrayLike: Fractional coordinate. A site equivalent to the
                original site, but on the other side of the slab
        """
    spg_analyzer = SpacegroupAnalyzer(self)
    ops = spg_analyzer.get_symmetry_operations(cartesian=cartesian)
    for op in ops:
        slab = self.copy()
        site_other = op.operate(point)
        if f'{site_other[2]:.6f}' == f'{point[2]:.6f}':
            continue
        slab.append('O', point, coords_are_cartesian=cartesian)
        slab.append('O', site_other, coords_are_cartesian=cartesian)
        if SpacegroupAnalyzer(slab).is_laue():
            break
        slab.remove_sites([len(slab) - 1])
        slab.remove_sites([len(slab) - 1])
    return site_other