from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_nearest_site(struct: Structure, coords: Sequence[float], site: PeriodicSite, r: float | None=None):
    """
    Given coords and a site, find closet site to coords.

    Args:
        coords (3x1 array): Cartesian coords of center of sphere
        site: site to find closest to coords
        r (float): radius of sphere. Defaults to diagonal of unit cell

    Returns:
        Closest site and distance.
    """
    index = struct.index(site)
    radius = r or np.linalg.norm(np.sum(struct.lattice.matrix, axis=0))
    ns = struct.get_sites_in_sphere(coords, radius, include_index=True)
    ns = [n for n in ns if n[2] == index]
    ns.sort(key=lambda x: x[1])
    return ns[0][0:2]