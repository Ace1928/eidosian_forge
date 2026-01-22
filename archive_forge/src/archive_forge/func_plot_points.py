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
def plot_points(points, lattice=None, coords_are_cartesian=False, fold=False, ax: plt.Axes=None, **kwargs):
    """Adds Points to a matplotlib Axes.

    Args:
        points: list of coordinates
        lattice: Lattice object used to convert from reciprocal to Cartesian coordinates
        coords_are_cartesian: Set to True if you are providing
            coordinates in Cartesian coordinates. Defaults to False.
            Requires lattice if False.
        fold: whether the points should be folded inside the first Brillouin Zone.
            Defaults to False. Requires lattice if True.
        ax: matplotlib Axes or None if a new figure should be created.
        kwargs: kwargs passed to the matplotlib function 'scatter'. Color defaults to blue

    Returns:
        matplotlib figure and matplotlib ax
    """
    ax, fig = get_ax3d_fig(ax)
    if 'color' not in kwargs:
        kwargs['color'] = 'b'
    if (not coords_are_cartesian or fold) and lattice is None:
        raise ValueError('coords_are_cartesian False or fold True require the lattice')
    for p in points:
        if fold:
            p = fold_point(p, lattice, coords_are_cartesian=coords_are_cartesian)
        elif not coords_are_cartesian:
            p = lattice.get_cartesian_coords(p)
        ax.scatter(*p, **kwargs)
    return (fig, ax)