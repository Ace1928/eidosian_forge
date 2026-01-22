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
def get_tasker2_slabs(self, tol: float=0.01, same_species_only: bool=True) -> list[Slab]:
    """Get a list of slabs that have been Tasker 2 corrected.

        Args:
            tol (float): Fractional tolerance to determine if atoms are within same plane.
            same_species_only (bool): If True, only those are of the exact same
                species as the atom at the outermost surface are considered for moving.
                Otherwise, all atoms regardless of species within tol are considered for moving.
                Default is True (usually the desired behavior).

        Returns:
            list[Slab]: Tasker 2 corrected slabs.
        """

    def get_equi_index(site: PeriodicSite) -> int:
        """Get the index of the equivalent site for a given site."""
        for idx, equi_sites in enumerate(symm_structure.equivalent_sites):
            if site in equi_sites:
                return idx
        raise ValueError('Cannot determine equi index!')
    sites = list(self.sites)
    slabs = []
    sorted_csites = sorted(sites, key=lambda site: site.c)
    n_layers_total = int(round(self.lattice.c / self.oriented_unit_cell.lattice.c))
    n_layers_slab = int(round((sorted_csites[-1].c - sorted_csites[0].c) * n_layers_total))
    slab_ratio = n_layers_slab / n_layers_total
    spg_analyzer = SpacegroupAnalyzer(self)
    symm_structure = spg_analyzer.get_symmetrized_structure()
    for surface_site, shift in [(sorted_csites[0], slab_ratio), (sorted_csites[-1], -slab_ratio)]:
        to_move = []
        fixed = []
        for site in sites:
            if abs(site.c - surface_site.c) < tol and (not same_species_only or site.species == surface_site.species):
                to_move.append(site)
            else:
                fixed.append(site)
        to_move = sorted(to_move, key=get_equi_index)
        grouped = [list(sites) for k, sites in itertools.groupby(to_move, key=get_equi_index)]
        if len(to_move) == 0 or any((len(g) % 2 != 0 for g in grouped)):
            warnings.warn('Odd number of sites to divide! Try changing the tolerance to ensure even division of sites or create supercells in a or b directions to allow for atoms to be moved!')
            continue
        combinations = []
        for g in grouped:
            combinations.append(list(itertools.combinations(g, int(len(g) / 2))))
        for selection in itertools.product(*combinations):
            species = [site.species for site in fixed]
            frac_coords = [site.frac_coords for site in fixed]
            for struct_matcher in to_move:
                species.append(struct_matcher.species)
                for group in selection:
                    if struct_matcher in group:
                        frac_coords.append(struct_matcher.frac_coords)
                        break
                else:
                    frac_coords.append(struct_matcher.frac_coords + [0, 0, shift])
            sp_fcoord = sorted(zip(species, frac_coords), key=lambda x: x[0])
            species = [x[0] for x in sp_fcoord]
            frac_coords = [x[1] for x in sp_fcoord]
            slab = Slab(self.lattice, species, frac_coords, self.miller_index, self.oriented_unit_cell, self.shift, self.scale_factor, energy=self.energy, reorient_lattice=self.reorient_lattice)
            slabs.append(slab)
    struct_matcher = StructureMatcher()
    return [ss[0] for ss in struct_matcher.group_structures(slabs)]