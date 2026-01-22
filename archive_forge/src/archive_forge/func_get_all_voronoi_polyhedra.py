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
def get_all_voronoi_polyhedra(self, structure: Structure):
    """Get the Voronoi polyhedra for all site in a simulation cell.

        Args:
            structure (Structure): Structure to be evaluated

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
    if len(structure) == 1:
        return [self.get_voronoi_polyhedra(structure, 0)]
    targets = structure.elements if self.targets is None else self.targets
    sites = [x.to_unit_cell() for x in structure]
    indices = [(idx, 0, 0, 0) for idx in range(len(structure))]
    all_neighs = structure.get_all_neighbors(self.cutoff, include_index=True, include_image=True)
    for neighs in all_neighs:
        sites.extend([x[0] for x in neighs])
        indices.extend([(x[2],) + x[3] for x in neighs])
    indices = np.array(indices, dtype=int)
    indices, uniq_inds = np.unique(indices, return_index=True, axis=0)
    sites = [sites[idx] for idx in uniq_inds]
    root_images, = np.nonzero(np.abs(indices[:, 1:]).max(axis=1) == 0)
    del indices
    qvoronoi_input = [s.coords for s in sites]
    voro = Voronoi(qvoronoi_input)
    return [self._extract_cell_info(idx, sites, targets, voro, self.compute_adj_neighbors) for idx in root_images.tolist()]