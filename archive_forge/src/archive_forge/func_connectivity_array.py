from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@property
def connectivity_array(self):
    """
        Provides connectivity array.

        Returns:
            connectivity: An array of shape [atom_i, atom_j, image_j]. atom_i is
            the index of the atom in the input structure. Since the second
            atom can be outside of the unit cell, it must be described
            by both an atom index and an image index. Array data is the
            solid angle of polygon between atom_i and image_j of atom_j
        """
    cart_coords = np.array(self.structure.cart_coords)
    all_sites = cart_coords[:, None, :] + self.cart_offsets[None, :, :]
    vt = Voronoi(all_sites.reshape((-1, 3)))
    n_images = all_sites.shape[1]
    cs = (len(self.structure), len(self.structure), len(self.cart_offsets))
    connectivity = np.zeros(cs)
    vts = np.array(vt.vertices)
    for (ki, kj), v in vt.ridge_dict.items():
        atom_i = ki // n_images
        atom_j = kj // n_images
        image_i = ki % n_images
        image_j = kj % n_images
        if image_i != n_images // 2 and image_j != n_images // 2:
            continue
        if image_i == n_images // 2:
            val = solid_angle(vt.points[ki], vts[v])
            connectivity[atom_i, atom_j, image_j] = val
        if image_j == n_images // 2:
            val = solid_angle(vt.points[kj], vts[v])
            connectivity[atom_j, atom_i, image_i] = val
        if -10.101 in vts[v]:
            warn('Found connectivity with infinite vertex. Cutoff is too low, and results may be incorrect')
    return connectivity