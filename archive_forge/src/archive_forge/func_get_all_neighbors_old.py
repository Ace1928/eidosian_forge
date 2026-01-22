from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
@deprecated(get_all_neighbors, 'This is retained purely for checking purposes.')
def get_all_neighbors_old(self, r, include_index=False, include_image=False, include_site=True):
    """Get neighbors for each atom in the unit cell, out to a distance r
        Returns a list of list of neighbors for each site in structure.
        Use this method if you are planning on looping over all sites in the
        crystal. If you only want neighbors for a particular site, use the
        method get_neighbors as it may not have to build such a large supercell
        However if you are looping over all sites in the crystal, this method
        is more efficient since it only performs one pass over a large enough
        supercell to contain all possible atoms out to a distance r.
        The return type is a [(site, dist) ...] since most of the time,
        subsequent processing requires the distance.

        A note about periodic images: Before computing the neighbors, this
        operation translates all atoms to within the unit cell (having
        fractional coordinates within [0,1)). This means that the "image" of a
        site does not correspond to how much it has been translates from its
        current position, but which image of the unit cell it resides.

        Args:
            r (float): Radius of sphere.
            include_index (bool): Whether to include the non-supercell site
                in the returned data
            include_image (bool): Whether to include the supercell image
                in the returned data
            include_site (bool): Whether to include the site in the returned
                data. Defaults to True.

        Returns:
            PeriodicNeighbor
        """
    recp_len = np.array(self.lattice.reciprocal_lattice.abc)
    maxr = np.ceil((r + 0.15) * recp_len / (2 * math.pi))
    nmin = np.floor(np.min(self.frac_coords, axis=0)) - maxr
    nmax = np.ceil(np.max(self.frac_coords, axis=0)) + maxr
    all_ranges = list(itertools.starmap(np.arange, zip(nmin, nmax)))
    lattice = self._lattice
    matrix = lattice.matrix
    neighbors = [[] for _ in range(len(self))]
    all_fcoords = np.mod(self.frac_coords, 1)
    coords_in_cell = np.dot(all_fcoords, matrix)
    site_coords = self.cart_coords
    indices = np.arange(len(self))
    for image in itertools.product(*all_ranges):
        coords = np.dot(image, matrix) + coords_in_cell
        all_dists = all_distances(coords, site_coords)
        all_within_r = np.bitwise_and(all_dists <= r, all_dists > 1e-08)
        for j, d, within_r in zip(indices, all_dists, all_within_r):
            if include_site:
                nnsite = PeriodicSite(self[j].species, coords[j], lattice, properties=self[j].properties, coords_are_cartesian=True, skip_checks=True)
            for i in indices[within_r]:
                item = []
                if include_site:
                    item.append(nnsite)
                item.append(d[i])
                if include_index:
                    item.append(j)
                if include_image:
                    item.append(image)
                neighbors[i].append(item)
    return neighbors