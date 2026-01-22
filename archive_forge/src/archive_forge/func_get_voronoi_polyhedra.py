from __future__ import annotations
import json
import math
import os
import warnings
from bisect import bisect_left
from collections import defaultdict, namedtuple
from copy import deepcopy
from functools import lru_cache
from math import acos, asin, atan2, cos, exp, fabs, pi, pow, sin, sqrt
from typing import TYPE_CHECKING, Any, Literal, get_args
import numpy as np
from monty.dev import deprecated, requires
from monty.serialization import loadfn
from ruamel.yaml import YAML
from scipy.spatial import Voronoi
from pymatgen.analysis.bond_valence import BV_PARAMS, BVAnalyzer
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, IStructure, PeriodicNeighbor, PeriodicSite, Site, Species, Structure
def get_voronoi_polyhedra(self, structure: Structure, n: int):
    """
        Gives a weighted polyhedra around a site.

        See ref: A Proposed Rigorous Definition of Coordination Number,
        M. O'Keeffe, Acta Cryst. (1979). A35, 772-775

        Args:
            structure (Structure): structure for which to evaluate the
                coordination environment.
            n (int): site index.

        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """
    targets = structure.elements if self.targets is None else self.targets
    center = structure[n]
    corners = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    d_corners = [np.linalg.norm(structure.lattice.get_cartesian_coords(c)) for c in corners]
    max_cutoff = max(d_corners) + 0.01
    while True:
        try:
            neighbors = structure.get_sites_in_sphere(center.coords, self.cutoff)
            neighbors = [ngbr[0] for ngbr in sorted(neighbors, key=lambda s: s[1])]
            qvoronoi_input = [site.coords for site in neighbors]
            voro = Voronoi(qvoronoi_input)
            cell_info = self._extract_cell_info(0, neighbors, targets, voro, self.compute_adj_neighbors)
            break
        except RuntimeError as exc:
            if self.cutoff >= max_cutoff:
                if exc.args and 'vertex' in exc.args[0]:
                    raise exc
                raise RuntimeError('Error in Voronoi neighbor finding; max cutoff exceeded')
            self.cutoff = min(self.cutoff * 2, max_cutoff + 0.001)
    return cell_info