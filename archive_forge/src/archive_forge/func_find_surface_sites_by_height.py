from __future__ import annotations
import itertools
import os
from typing import TYPE_CHECKING
import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from monty.serialization import loadfn
from scipy.spatial import Delaunay
from pymatgen import vis
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
def find_surface_sites_by_height(self, slab: Slab, height=0.9, xy_tol=0.05):
    """This method finds surface sites by determining which sites are
        within a threshold value in height from the topmost site in a list of
        sites.

        Args:
            slab (Slab): slab for which to find surface sites
            height (float): threshold in angstroms of distance from topmost
                site in slab along the slab c-vector to include in surface
                site determination
            xy_tol (float): if supplied, will remove any sites which are
                within a certain distance in the miller plane.

        Returns:
            list of sites selected to be within a threshold of the highest
        """
    m_projs = np.array([np.dot(site.coords, self.mvec) for site in slab])
    mask = m_projs - np.amax(m_projs) >= -height
    surf_sites = [slab.sites[n] for n in np.where(mask)[0]]
    if xy_tol:
        surf_sites = [s for h, s in zip(m_projs[mask], surf_sites)]
        surf_sites.reverse()
        unique_sites: list = []
        unique_perp_fracs: list = []
        for site in surf_sites:
            this_perp = site.coords - np.dot(site.coords, self.mvec)
            this_perp_frac = slab.lattice.get_fractional_coords(this_perp)
            if not in_coord_list_pbc(unique_perp_fracs, this_perp_frac):
                unique_sites.append(site)
                unique_perp_fracs.append(this_perp_frac)
        surf_sites = unique_sites
    return surf_sites