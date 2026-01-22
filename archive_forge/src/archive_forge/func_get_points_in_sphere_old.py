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
@deprecated(get_points_in_sphere, 'This is retained purely for checking purposes.')
def get_points_in_sphere_old(self, frac_points: ArrayLike, center: ArrayLike, r: float, zip_results=True) -> list[tuple[np.ndarray, float, int, np.ndarray]] | tuple[list[np.ndarray], list[float], list[int], list[np.ndarray]]:
    """Find all points within a sphere from the point taking into account
        periodic boundary conditions. This includes sites in other periodic
        images. Does not support partial periodic boundary conditions.

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
    if self.pbc != (True, True, True):
        raise RuntimeError('get_points_in_sphere_old does not support partial periodic boundary conditions')
    recp_len = np.array(self.reciprocal_lattice.abc) / (2 * np.pi)
    nmax = float(r) * recp_len + 0.01
    pcoords = self.get_fractional_coords(center)
    center = np.array(center)
    n = len(frac_points)
    frac_coords = np.array(frac_points) % 1
    indices = np.arange(n)
    mins = np.floor(pcoords - nmax)
    maxes = np.ceil(pcoords + nmax)
    arange = np.arange(start=mins[0], stop=maxes[0], dtype=int)
    brange = np.arange(start=mins[1], stop=maxes[1], dtype=int)
    crange = np.arange(start=mins[2], stop=maxes[2], dtype=int)
    arange = arange[:, None] * np.array([1, 0, 0], dtype=int)[None, :]
    brange = brange[:, None] * np.array([0, 1, 0], dtype=int)[None, :]
    crange = crange[:, None] * np.array([0, 0, 1], dtype=int)[None, :]
    images = arange[:, None, None] + brange[None, :, None] + crange[None, None, :]
    shifted_coords = frac_coords[:, None, None, None, :] + images[None, :, :, :, :]
    cart_coords = self.get_cartesian_coords(frac_coords)
    cart_images = self.get_cartesian_coords(images)
    coords = cart_coords[:, None, None, None, :] + cart_images[None, :, :, :, :]
    coords -= center[None, None, None, None, :]
    coords **= 2
    d_2 = np.sum(coords, axis=4)
    within_r = np.where(d_2 <= r ** 2)
    if zip_results:
        return list(zip(shifted_coords[within_r], np.sqrt(d_2[within_r]), indices[within_r[0]], images[within_r[1:]]))
    return (shifted_coords[within_r], np.sqrt(d_2[within_r]), indices[within_r[0]], images[within_r[1:]])