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
def get_surface_sites(self, tag: bool=False) -> dict[str, list]:
    """Returns the surface sites and their indices in a dictionary.
        Useful for analysis involving broken bonds and for finding adsorption sites.

        The oriented unit cell of the slab will determine the
        coordination number of a typical site.
        We use VoronoiNN to determine the coordination number of sites.
        Due to the pathological error resulting from some surface sites in the
        VoronoiNN, we assume any site that has this error is a surface
        site as well. This will only work for single-element systems for now.

        Args:
            tag (bool): Option to adds site attribute "is_surfsite" (bool)
                to all sites of slab. Defaults to False

        Returns:
            A dictionary grouping sites on top and bottom of the slab together.
                {"top": [sites with indices], "bottom": [sites with indices]}

        Todo:
            Is there a way to determine site equivalence between sites in a slab
            and bulk system? This would allow us get the coordination number of
            a specific site for multi-elemental systems or systems with more
            than one inequivalent site. This will allow us to use this for
            compound systems.
        """
    from pymatgen.analysis.local_env import VoronoiNN
    spg_analyzer = SpacegroupAnalyzer(self.oriented_unit_cell)
    u_cell = spg_analyzer.get_symmetrized_structure()
    cn_dict: dict = {}
    voronoi_nn = VoronoiNN()
    unique_indices = [equ[0] for equ in u_cell.equivalent_indices]
    for idx in unique_indices:
        el = u_cell[idx].species_string
        if el not in cn_dict:
            cn_dict[el] = []
        cn = voronoi_nn.get_cn(u_cell, idx, use_weights=True)
        cn = float(f'{round(cn, 5):.5f}')
        if cn not in cn_dict[el]:
            cn_dict[el].append(cn)
    voronoi_nn = VoronoiNN()
    surf_sites_dict: dict = {'top': [], 'bottom': []}
    properties: list = []
    for idx, site in enumerate(self):
        is_top: bool = site.frac_coords[2] > self.center_of_mass[2]
        try:
            cn = float(f'{round(voronoi_nn.get_cn(self, idx, use_weights=True), 5):.5f}')
            if cn < min(cn_dict[site.species_string]):
                properties.append(True)
                key = 'top' if is_top else 'bottom'
                surf_sites_dict[key].append([site, idx])
            else:
                properties.append(False)
        except RuntimeError:
            properties.append(True)
            key = 'top' if is_top else 'bottom'
            surf_sites_dict[key].append([site, idx])
    if tag:
        self.add_site_property('is_surf_site', properties)
    return surf_sites_dict