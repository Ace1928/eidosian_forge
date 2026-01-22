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
def repair_broken_bonds(self, slab: Slab, bonds: dict[tuple[Species | Element, Species | Element], float]) -> Slab:
    """Repair broken bonds (specified by the bonds parameter) due to
        slab cleaving, and repair them by moving undercoordinated atoms
        to the other surface.

        How it works:
            For example a P-O4 bond may have P and O(4-x) on one side
            of the surface, and Ox on the other side, this method would
            first move P (the reference atom) to the other side,
            find its missing nearest neighbours (Ox), and move P
            and Ox back together.

        Args:
            slab (Slab): The Slab to repair.
            bonds (dict): A {(species1, species2): max_bond_dist} dict.
                For example, PO4 groups may be defined as {("P", "O"): 3}.

        Returns:
            Slab: The repaired Slab.
        """
    for species_pair, bond_dist in bonds.items():
        cn_dict = {}
        for idx, ele in enumerate(species_pair):
            cn_list = []
            for site in self.oriented_unit_cell:
                ref_cn = 0
                if site.species_string == ele:
                    for nn in self.oriented_unit_cell.get_neighbors(site, bond_dist):
                        if nn[0].species_string == species_pair[idx - 1]:
                            ref_cn += 1
                cn_list.append(ref_cn)
            cn_dict[ele] = cn_list
        if max(cn_dict[species_pair[0]]) > max(cn_dict[species_pair[1]]):
            ele_ref, ele_other = species_pair
        else:
            ele_other, ele_ref = species_pair
        for idx, site in enumerate(slab):
            if site.species_string == ele_ref:
                ref_cn = sum((1 if neighbor.species_string == ele_other else 0 for neighbor in slab.get_neighbors(site, bond_dist)))
                if ref_cn not in cn_dict[ele_ref]:
                    slab = self.move_to_other_side(slab, [idx])
                    neighbors = slab.get_neighbors(slab[idx], r=bond_dist)
                    to_move = [nn[2] for nn in neighbors if nn[0].species_string == ele_other]
                    to_move.append(idx)
                    slab = self.move_to_other_side(slab, to_move)
    return slab