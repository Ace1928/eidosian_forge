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
def nonstoichiometric_symmetrized_slab(self, init_slab: Slab) -> list[Slab]:
    """Symmetrize the two surfaces of a Slab, but may break the stoichiometry.

        How it works:
            1. Check whether two surfaces of the slab are equivalent.
            If the point group of the slab has an inversion symmetry (
            ie. belong to one of the Laue groups), then it's assumed that the
            surfaces are equivalent.

            2.If not symmetrical, sites at the bottom of the slab will be removed
            until the slab is symmetric, which may break the stoichiometry.

        Args:
            init_slab (Slab): The initial Slab.

        Returns:
            list[Slabs]: The symmetrized Slabs.
        """
    if init_slab.is_symmetric():
        return [init_slab]
    non_stoich_slabs = []
    for surface in ('top', 'bottom'):
        is_sym: bool = False
        slab = init_slab.copy()
        slab.energy = init_slab.energy
        while not is_sym:
            z_coords: list[float] = [site[2] for site in slab.frac_coords]
            if surface == 'top':
                slab.remove_sites([z_coords.index(max(z_coords))])
            else:
                slab.remove_sites([z_coords.index(min(z_coords))])
            if len(slab) <= len(self.parent):
                warnings.warn('Too many sites removed, please use a larger slab.')
                break
            if slab.is_symmetric():
                is_sym = True
                non_stoich_slabs.append(slab)
    return non_stoich_slabs