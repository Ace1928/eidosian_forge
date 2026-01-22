from __future__ import annotations
import copy
import itertools
import logging
import math
import typing
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, cast, no_type_check
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import palettable
import scipy.interpolate as scint
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from monty.dev import requires
from monty.json import jsanitize
from pymatgen.core import Element
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.util.plotting import add_fig_kwargs, get_ax3d_fig, pretty_plot
def plot_ellipsoid(hessian, center, lattice=None, rescale=1.0, ax: plt.Axes=None, coords_are_cartesian=False, arrows=False, **kwargs):
    """Plots a 3D ellipsoid rappresenting the Hessian matrix in input.
    Useful to get a graphical visualization of the effective mass
    of a band in a single k-point.

    Args:
        hessian: the Hessian matrix
        center: the center of the ellipsoid in reciprocal coords (Default)
        lattice: Lattice object of the Brillouin zone
        rescale: factor for size scaling of the ellipsoid
        ax: matplotlib Axes or None if a new figure should be created.
        coords_are_cartesian: Set to True if you are providing a center in
            Cartesian coordinates. Defaults to False.
        arrows: whether to plot arrows for the principal axes of the ellipsoid. Defaults to False.
        **kwargs: passed to the matplotlib function 'plot_wireframe'.
            Color defaults to blue, rstride and cstride
            default to 4, alpha defaults to 0.2.

    Returns:
        matplotlib figure and matplotlib ax

    Example of use:
        fig,ax=plot_wigner_seitz(struct.reciprocal_lattice)
        plot_ellipsoid(hessian,[0.0,0.0,0.0], struct.reciprocal_lattice,ax=ax)
    """
    if not coords_are_cartesian and lattice is None:
        raise ValueError('coords_are_cartesian False or fold True require the lattice')
    if not coords_are_cartesian:
        center = lattice.get_cartesian_coords(center)
    if 'color' not in kwargs:
        kwargs['color'] = 'b'
    if 'rstride' not in kwargs:
        kwargs['rstride'] = 4
    if 'cstride' not in kwargs:
        kwargs['cstride'] = 4
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.2
    _U, s, rotation = np.linalg.svd(hessian)
    radii = 1.0 / np.sqrt(s)
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) * rescale + center
    ax, fig = get_ax3d_fig(ax)
    ax.plot_wireframe(x, y, z, **kwargs)
    if arrows:
        color = ('b', 'g', 'r')
        em = np.zeros((3, 3))
        for i in range(3):
            em[i, :] = rotation[i, :] / np.linalg.norm(rotation[i, :])
        for i in range(3):
            ax.quiver3D(center[0], center[1], center[2], em[i, 0], em[i, 1], em[i, 2], pivot='tail', arrow_length_ratio=0.2, length=radii[i] * rescale, color=color[i])
    return (fig, ax)