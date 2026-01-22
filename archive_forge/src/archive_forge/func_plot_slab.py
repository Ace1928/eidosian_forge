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
def plot_slab(slab: Slab, ax: plt.Axes, scale=0.8, repeat=5, window=1.5, draw_unit_cell=True, decay=0.2, adsorption_sites=True, inverse=False):
    """Function that helps visualize the slab in a 2-D plot, for convenient
    viewing of output of AdsorbateSiteFinder.

    Args:
        slab (slab): Slab object to be visualized
        ax (axes): matplotlib axes with which to visualize
        scale (float): radius scaling for sites
        repeat (int): number of repeating unit cells to visualize
        window (float): window for setting the axes limits, is essentially
            a fraction of the unit cell limits
        draw_unit_cell (bool): flag indicating whether or not to draw cell
        decay (float): how the alpha-value decays along the z-axis
        inverse (bool): invert z axis to plot opposite surface
    """
    orig_slab = slab.copy()
    slab = reorient_z(slab)
    orig_cell = slab.lattice.matrix.copy()
    if repeat:
        slab.make_supercell([repeat, repeat, 1])
    coords = np.array(sorted(slab.cart_coords, key=lambda x: x[2]))
    sites = sorted(slab.sites, key=lambda x: x.coords[2])
    alphas = 1 - decay * (np.max(coords[:, 2]) - coords[:, 2])
    alphas = alphas.clip(min=0)
    corner = [0, 0, slab.lattice.get_fractional_coords(coords[-1])[-1]]
    corner = slab.lattice.get_cartesian_coords(corner)[:2]
    vertices = orig_cell[:2, :2]
    lattice_sum = vertices[0] + vertices[1]
    if inverse:
        alphas = np.array(reversed(alphas))
        sites = list(reversed(sites))
        coords = np.array(reversed(coords))
    for n, coord in enumerate(coords):
        radius = sites[n].species.elements[0].atomic_radius * scale
        ax.add_patch(patches.Circle(coord[:2] - lattice_sum * (repeat // 2), radius, color='w', zorder=2 * n))
        color = color_dict[sites[n].species.elements[0].symbol]
        ax.add_patch(patches.Circle(coord[:2] - lattice_sum * (repeat // 2), radius, facecolor=color, alpha=alphas[n], edgecolor='k', lw=0.3, zorder=2 * n + 1))
    if adsorption_sites:
        asf = AdsorbateSiteFinder(orig_slab)
        if inverse:
            inverse_slab = orig_slab.copy()
            inverse_slab.make_supercell([1, 1, -1])
            asf = AdsorbateSiteFinder(inverse_slab)
        ads_sites = asf.find_adsorption_sites()['all']
        symm_op = get_rot(orig_slab)
        ads_sites = [symm_op.operate(ads_site)[:2].tolist() for ads_site in ads_sites]
        ax.plot(*zip(*ads_sites), color='k', marker='x', markersize=10, mew=1, linestyle='', zorder=10000)
    if draw_unit_cell:
        vertices = np.insert(vertices, 1, lattice_sum, axis=0).tolist()
        vertices += [[0.0, 0.0]]
        vertices = [[0.0, 0.0], *vertices]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        vertices = [(np.array(vert) + corner).tolist() for vert in vertices]
        path = Path(vertices, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.5, zorder=2 * n + 2)
        ax.add_patch(patch)
    ax.set_aspect('equal')
    center = corner + lattice_sum / 2.0
    extent = np.max(lattice_sum)
    lim_array = [center - extent * window, center + extent * window]
    x_lim = [ele[0] for ele in lim_array]
    y_lim = [ele[1] for ele in lim_array]
    ax.set(xlim=x_lim, ylim=y_lim)
    return ax