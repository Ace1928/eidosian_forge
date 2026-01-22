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
def get_points_in_sphere_py(self, frac_points: ArrayLike, center: ArrayLike, r: float, zip_results=True) -> list[tuple[np.ndarray, float, int, np.ndarray]] | list[np.ndarray]:
    """Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic
        images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelepiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1"s it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            frac_points: All points in the lattice in fractional coordinates.
            center: Cartesian coordinates of center of sphere.
            r: radius of sphere.
            zip_results (bool): Whether to zip the results together to group by
                point, or return the raw fcoord, dist, index arrays

        Returns:
            if zip_results:
                [(fcoord, dist, index, supercell_image) ...] since most of the time, subsequent
                processing requires the distance, index number of the atom, or index of the image
            else:
                frac_coords, dists, inds, image
        """
    cart_coords = self.get_cartesian_coords(frac_points)
    neighbors = get_points_in_spheres(all_coords=cart_coords, center_coords=np.array([center]), r=r, pbc=self.pbc, numerical_tol=1e-08, lattice=self, return_fcoords=True)[0]
    if len(neighbors) < 1:
        return [] if zip_results else [()] * 4
    if zip_results:
        return neighbors
    return [np.array(i) for i in list(zip(*neighbors))]