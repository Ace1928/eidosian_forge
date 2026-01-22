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
def fit_anonymous(self, struct1: Structure, struct2: Structure, niggli: bool=True, skip_structure_reduction: bool=False) -> bool:
    """
        Performs an anonymous fitting, which allows distinct species in one structure to map
        to another. E.g., to compare if the Li2O and Na2O structures are similar.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure
            niggli (bool): If true, perform Niggli reduction for struct1 and struct2
            skip_structure_reduction (bool): Defaults to False
                If True, skip to get a primitive structure and perform Niggli reduction for struct1 and struct2

        Returns:
            bool: Whether a species mapping can map struct1 to struct2
        """
    struct1, struct2 = self._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = self._preprocess(struct1, struct2, niggli, skip_structure_reduction)
    matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, break_on_match=True, single_match=True)
    return bool(matches)