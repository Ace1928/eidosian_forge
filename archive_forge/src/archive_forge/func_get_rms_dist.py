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
def get_rms_dist(self, struct1, struct2):
    """
        Calculate RMS displacement between two structures.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            rms displacement normalized by (Vol / nsites) ** (1/3)
            and maximum distance between paired sites. If no matching
            lattice is found None is returned.
        """
    struct1, struct2 = self._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = self._preprocess(struct1, struct2)
    match = self._match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False)
    if match is None:
        return None
    return (match[0], max(match[1]))