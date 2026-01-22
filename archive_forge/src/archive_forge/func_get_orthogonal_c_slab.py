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
def get_orthogonal_c_slab(self) -> Slab:
    """Generate a Slab where the normal (c lattice vector) is
        forced to be orthogonal to the surface a and b lattice vectors.

        **Note that this breaks inherent symmetries in the slab.**

        It should be pointed out that orthogonality is not required to get good
        surface energies, but it can be useful in cases where the slabs are
        subsequently used for postprocessing of some kind, e.g. generating
        grain boundaries or interfaces.
        """
    a, b, c = self.lattice.matrix
    _new_c = np.cross(a, b)
    _new_c /= np.linalg.norm(_new_c)
    new_c = np.dot(c, _new_c) * _new_c
    new_latt = Lattice([a, b, new_c])
    return Slab(lattice=new_latt, species=self.species_and_occu, coords=self.cart_coords, miller_index=self.miller_index, oriented_unit_cell=self.oriented_unit_cell, shift=self.shift, scale_factor=self.scale_factor, coords_are_cartesian=True, energy=self.energy, reorient_lattice=self.reorient_lattice, site_properties=self.site_properties)