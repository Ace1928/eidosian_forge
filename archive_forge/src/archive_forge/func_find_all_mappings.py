from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def find_all_mappings(self, other_lattice: Lattice, ltol: float=1e-05, atol: float=1, skip_rotation_matrix: bool=False) -> Iterator[tuple[Lattice, np.ndarray | None, np.ndarray]]:
    """Finds all mappings between current lattice and another lattice.

        Args:
            other_lattice (Lattice): Another lattice that is equivalent to this one.
            ltol (float): Tolerance for matching lengths. Defaults to 1e-5.
            atol (float): Tolerance for matching angles. Defaults to 1.
            skip_rotation_matrix (bool): Whether to skip calculation of the
                rotation matrix

        Yields:
            (aligned_lattice, rotation_matrix, scale_matrix) if a mapping is
            found. aligned_lattice is a rotated version of other_lattice that
            has the same lattice parameters, but which is aligned in the
            coordinate system of this lattice so that translational points
            match up in 3D. rotation_matrix is the rotation that has to be
            applied to other_lattice to obtain aligned_lattice, i.e.,
            aligned_matrix = np.inner(other_lattice, rotation_matrix) and
            op = SymmOp.from_rotation_and_translation(rotation_matrix)
            aligned_matrix = op.operate_multi(latt.matrix)
            Finally, scale_matrix is the integer matrix that expresses
            aligned_matrix as a linear combination of this
            lattice, i.e., aligned_matrix = np.dot(scale_matrix, self.matrix)

            None is returned if no matches are found.
        """
    lengths = other_lattice.lengths
    alpha, beta, gamma = other_lattice.angles
    frac, dist, _, _ = self.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max(lengths) * (1 + ltol), zip_results=False)
    cart = self.get_cartesian_coords(frac)
    inds = [np.logical_and(dist / ln < 1 + ltol, dist / ln > 1 / (1 + ltol)) for ln in lengths]
    c_a, c_b, c_c = (cart[i] for i in inds)
    f_a, f_b, f_c = (frac[i] for i in inds)
    l_a, l_b, l_c = (np.sum(c ** 2, axis=-1) ** 0.5 for c in (c_a, c_b, c_c))

    def get_angles(v1, v2, l1, l2):
        x = np.inner(v1, v2) / l1[:, None] / l2
        x[x > 1] = 1
        x[x < -1] = -1
        return np.arccos(x) * 180.0 / np.pi
    alpha_b = np.abs(get_angles(c_b, c_c, l_b, l_c) - alpha) < atol
    beta_b = np.abs(get_angles(c_a, c_c, l_a, l_c) - beta) < atol
    gamma_b = np.abs(get_angles(c_a, c_b, l_a, l_b) - gamma) < atol
    for idx, all_j in enumerate(gamma_b):
        inds = np.logical_and(all_j[:, None], np.logical_and(alpha_b, beta_b[idx][None, :]))
        for j, k in np.argwhere(inds):
            scale_m = np.array((f_a[idx], f_b[j], f_c[k]), dtype=int)
            if abs(np.linalg.det(scale_m)) < 1e-08:
                continue
            aligned_m = np.array((c_a[idx], c_b[j], c_c[k]))
            rotation_m = None if skip_rotation_matrix else np.linalg.solve(aligned_m, other_lattice.matrix)
            yield (Lattice(aligned_m), rotation_m, scale_m)