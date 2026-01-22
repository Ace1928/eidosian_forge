from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors

        Calculate the mapping from superset to subset.

        Args:
            superset (Structure): Structure containing at least the sites in
                subset (within the structure matching tolerance)
            subset (Structure): Structure containing some of the sites in
                superset (within the structure matching tolerance)

        Returns:
            numpy array such that superset.sites[mapping] is within matching
            tolerance of subset.sites or None if no such mapping is possible
        