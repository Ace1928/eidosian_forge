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
@classmethod
def from_bulk_and_miller(cls, structure, miller_index, min_slab_size=8.0, min_vacuum_size=10.0, max_normal_search=None, center_slab=True, selective_dynamics=False, undercoord_threshold=0.09) -> Self:
    """This method constructs the adsorbate site finder from a bulk
        structure and a miller index, which allows the surface sites to be
        determined from the difference in bulk and slab coordination, as
        opposed to the height threshold.

        Args:
            structure (Structure): structure from which slab
                input to the ASF is constructed
            miller_index (3-tuple or list): miller index to be used
            min_slab_size (float): min slab size for slab generation
            min_vacuum_size (float): min vacuum size for slab generation
            max_normal_search (int): max normal search for slab generation
            center_slab (bool): whether to center slab in slab generation
            selective dynamics (bool): whether to assign surface sites
                to selective dynamics
            undercoord_threshold (float): threshold of "undercoordation"
                to use for the assignment of surface sites. Default is
                0.1, for which surface sites will be designated if they
                are 10% less coordinated than their bulk counterpart
        """
    vnn_bulk = VoronoiNN(tol=0.05)
    bulk_coords = [len(vnn_bulk.get_nn(structure, n)) for n in range(len(structure))]
    struct = structure.copy(site_properties={'bulk_coordinations': bulk_coords})
    slabs = generate_all_slabs(struct, max_index=max(miller_index), min_slab_size=min_slab_size, min_vacuum_size=min_vacuum_size, max_normal_search=max_normal_search, center_slab=center_slab)
    slab_dict = {slab.miller_index: slab for slab in slabs}
    if miller_index not in slab_dict:
        raise ValueError('Miller index not in slab dict')
    this_slab = slab_dict[miller_index]
    vnn_surface = VoronoiNN(tol=0.05, allow_pathological=True)
    surf_props, under_coords = ([], [])
    this_mi_vec = get_mi_vec(this_slab)
    mi_mags = [np.dot(this_mi_vec, site.coords) for site in this_slab]
    average_mi_mag = np.average(mi_mags)
    for n, site in enumerate(this_slab):
        bulk_coord = this_slab.site_properties['bulk_coordinations'][n]
        slab_coord = len(vnn_surface.get_nn(this_slab, n))
        mi_mag = np.dot(this_mi_vec, site.coords)
        under_coord = (bulk_coord - slab_coord) / bulk_coord
        under_coords += [under_coord]
        if under_coord > undercoord_threshold and mi_mag > average_mi_mag:
            surf_props += ['surface']
        else:
            surf_props += ['subsurface']
    new_site_properties = {'surface_properties': surf_props, 'undercoords': under_coords}
    new_slab = this_slab.copy(site_properties=new_site_properties)
    return cls(new_slab, selective_dynamics)