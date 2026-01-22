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
def symmetrically_add_atom(self, species: str | Element | Species, point: ArrayLike, specie: str | Element | Species | None=None, coords_are_cartesian: bool=False) -> None:
    """Add a species at a specified site in a slab. Will also add an
        equivalent site on the other side of the slab to maintain symmetry.

        TODO (@DanielYang59): use "site" over "point" as arg name for consistency

        Args:
            species (str | Element | Species): The species to add.
            point (ArrayLike): The coordinate of the target site.
            specie: Deprecated argument name in #3691. Use 'species' instead.
            coords_are_cartesian (bool): If the site is in Cartesian coordinates.
        """
    if specie is not None:
        warnings.warn("The argument 'specie' is deprecated. Use 'species' instead.", DeprecationWarning)
        species = specie
    equi_site = self.get_symmetric_site(point, cartesian=coords_are_cartesian)
    self.append(species, point, coords_are_cartesian=coords_are_cartesian)
    self.append(species, equi_site, coords_are_cartesian=coords_are_cartesian)